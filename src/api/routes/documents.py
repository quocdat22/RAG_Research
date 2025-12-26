"""Document management routes."""
import logging
import shutil
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import settings
from src.api.schemas import (
    DocumentChunksResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentUploadResponse,
)
from src.embedding.embedder import get_embedder
from src.ingestion.chunking import smart_chunk_documents, smart_chunk_markdown
from src.ingestion.loaders import DocumentLoader
from src.storage.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Args:
        file: Uploaded file (PDF, DOCX, or TXT)
        
    Returns:
        Upload response with document ID and chunk count
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".txt"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Check if using Supabase
        use_supabase = settings.environment == "production" or settings.use_supabase_storage
        supabase_doc_id = None
        file_path = None
        temp_file = False
        
        if use_supabase and settings.supabase_url and settings.supabase_key:
            # Upload to Supabase Storage
            try:
                from src.storage.supabase_client import get_supabase_storage
                from datetime import datetime
                
                supabase_storage = get_supabase_storage()
                
                # Add timestamp to filename to avoid duplicates
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = Path(file.filename).stem, Path(file.filename).suffix
                unique_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
                
                doc_record = supabase_storage.upload_document(
                    file_path=unique_filename,
                    file_content=file_content,
                    metadata={
                        "file_type": file_extension,
                        "original_filename": file.filename
                    }
                )
                supabase_doc_id = doc_record['id']
                logger.info(f"✅ Uploaded to Supabase: {unique_filename} (ID: {supabase_doc_id})")
                
                # Create temp file for processing only
                file_path = settings.documents_dir / f"temp_{supabase_doc_id}_{file.filename}"
                with open(file_path, "wb") as f:
                    f.write(file_content)
                temp_file = True
                
            except Exception as e:
                logger.error(f"❌ Supabase upload failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload to Supabase: {str(e)}")
        else:
            # Save locally only if not using Supabase
            file_path = settings.documents_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"💾 Saved locally: {file.filename}")
        
        # Load document (returns pages and is_markdown flag)
        pages, is_markdown = DocumentLoader.load(file_path)
        
        # Extract metadata from first few pages
        from src.ingestion.metadata_extractor import get_metadata_extractor
        
        metadata_extractor = get_metadata_extractor()
        # Combine first 3 pages for metadata extraction
        first_pages_text = "\n\n".join([
            page.content for page in pages[:3]
        ])
        rich_metadata = metadata_extractor.extract(first_pages_text, file.filename)
        
        # Update document metadata with extracted rich metadata
        for page in pages:
            if page.metadata:
                page.metadata.authors = rich_metadata.get("authors")
                page.metadata.year = rich_metadata.get("year")
                page.metadata.keywords = rich_metadata.get("keywords")
                page.metadata.abstract = rich_metadata.get("abstract")
                page.metadata.doi = rich_metadata.get("doi")
                page.metadata.arxiv_id = rich_metadata.get("arxiv_id")
                page.metadata.venue = rich_metadata.get("venue")
        
        logger.info(f"Extracted metadata: {list(rich_metadata.keys())}")
        
        # Chunk document based on content type
        if is_markdown:
            # Use markdown-aware chunking that separates text and tables
            chunks = smart_chunk_markdown(
                pages,
                chunk_size=settings.chunking.size,
                chunk_overlap=settings.chunking.overlap
            )
            logger.info(f"Used LlamaParse + markdown chunking for {file.filename}")
        else:
            # Standard chunking for non-markdown content
            chunks = smart_chunk_documents(
                pages,
                chunk_size=settings.chunking.size,
                chunk_overlap=settings.chunking.overlap
            )
        
        # Generate embeddings
        embedder = get_embedder()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        
        # Store in vector database
        vector_store = get_vector_store()
        document_id = vector_store.add_documents(chunks, embeddings)
        
        # Save chunks to Supabase if using it
        if use_supabase and supabase_doc_id:
            try:
                chunk_data = [
                    {
                        "content": chunk.text,
                        "embedding_id": document_id,  # ChromaDB document ID
                        "metadata": chunk.metadata.dict() if hasattr(chunk.metadata, 'dict') else {}
                    }
                    for chunk in chunks
                ]
                supabase_storage.save_chunks(supabase_doc_id, chunk_data)
                logger.info(f"✅ Saved {len(chunks)} chunks to Supabase")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save chunks to Supabase: {e}")
        
        # Cleanup temp file
        if temp_file and file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"🗑️ Deleted temp file: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
        
        logger.info(
            f"✅ Processed document {file.filename}: "
            f"{len(chunks)} chunks, document_id={document_id}"
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunk_count=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of documents with metadata
    """
    try:
        # Check if using Supabase
        use_supabase = settings.environment == "production" or settings.use_supabase_storage
        
        if use_supabase and settings.supabase_url and settings.supabase_key:
            # Get documents from Supabase
            try:
                from src.storage.supabase_client import get_supabase_storage
                supabase_storage = get_supabase_storage()
                
                supabase_docs = supabase_storage.list_documents()
                
                # Convert Supabase format to DocumentInfo format
                doc_infos = []
                for doc in supabase_docs:
                    doc_info = DocumentInfo(
                        document_id=doc['id'],
                        filename=doc['filename'],
                        file_type=doc.get('file_type'),
                        upload_timestamp=doc.get('upload_date'),
                        authors=doc.get('metadata', {}).get('authors'),
                        year=doc.get('metadata', {}).get('year'),
                        keywords=doc.get('metadata', {}).get('keywords'),
                        abstract=doc.get('metadata', {}).get('abstract'),
                        doi=doc.get('metadata', {}).get('doi'),
                        arxiv_id=doc.get('metadata', {}).get('arxiv_id'),
                        venue=doc.get('metadata', {}).get('venue')
                    )
                    doc_infos.append(doc_info)
                
                logger.info(f"✅ Retrieved {len(doc_infos)} documents from Supabase")
                return DocumentListResponse(documents=doc_infos, total=len(doc_infos))
            except Exception as e:
                logger.error(f"❌ Failed to get documents from Supabase: {e}")
                # Fall back to ChromaDB
        
        # Get documents from ChromaDB (local)
        vector_store = get_vector_store()
        documents = vector_store.get_all_documents()
        
        # Documents already include rich metadata from get_all_documents()
        doc_infos = [DocumentInfo(**doc) for doc in documents]
        
        return DocumentListResponse(
            documents=doc_infos,
            total=len(doc_infos)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: Document ID to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        # Check if using Supabase
        use_supabase = settings.environment == "production" or settings.use_supabase_storage
        chunks_deleted = 0
        
        if use_supabase and settings.supabase_url and settings.supabase_key:
            # Delete from Supabase
            try:
                from src.storage.supabase_client import get_supabase_storage
                supabase_storage = get_supabase_storage()
                
                # Get chunk count before delete
                doc = supabase_storage.get_document(document_id)
                if not doc:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                chunks_deleted = doc.get('chunk_count', 0)
                
                # Delete from Supabase (includes storage file and chunks)
                supabase_storage.delete_document(document_id)
                logger.info(f"✅ Deleted document {document_id} from Supabase")
                
                return DocumentDeleteResponse(
                    document_id=document_id,
                    chunks_deleted=chunks_deleted
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Failed to delete from Supabase: {e}")
                # Fall back to ChromaDB
        
        # Delete from ChromaDB (local)
        vector_store = get_vector_store()
        chunks_deleted = vector_store.delete_document(document_id)
        
        if chunks_deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Deleted document {document_id}: {chunks_deleted} chunks")
        
        return DocumentDeleteResponse(
            document_id=document_id,
            chunks_deleted=chunks_deleted
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str):
    """
    Get all chunks for a specific document.
    
    Args:
        document_id: Document ID to retrieve chunks for
        
    Returns:
        List of chunks with text and metadata
    """
    try:
        # Check if using Supabase
        use_supabase = settings.environment == "production" or settings.use_supabase_storage
        
        if use_supabase and settings.supabase_url and settings.supabase_key:
            # Get chunks from Supabase
            try:
                from src.storage.supabase_client import get_supabase_storage
                supabase_storage = get_supabase_storage()
                
                supabase_chunks = supabase_storage.get_document_chunks(document_id)
                
                if not supabase_chunks:
                    raise HTTPException(status_code=404, detail="Document not found or has no chunks")
                
                # Convert to expected format
                chunks = []
                for chunk in supabase_chunks:
                    chunks.append({
                        "chunk_id": chunk['id'],
                        "text": chunk['content'],
                        "metadata": chunk.get('metadata', {})
                    })
                
                logger.info(f"✅ Retrieved {len(chunks)} chunks from Supabase for document {document_id}")
                return DocumentChunksResponse(
                    document_id=document_id,
                    chunks=chunks,
                    total=len(chunks)
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Failed to get chunks from Supabase: {e}")
                # Fall back to ChromaDB
        
        # Get chunks from ChromaDB (local)
        vector_store = get_vector_store()
        chunks = vector_store.get_document_chunks(document_id)
        
        if not chunks:
            # Check if document exists at all (might have 0 chunks or wrong ID)
            all_docs = vector_store.get_all_documents()
            doc_exists = any(d["document_id"] == document_id for d in all_docs)
            if not doc_exists:
                raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentChunksResponse(
            document_id=document_id,
            chunks=chunks,
            total=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/metadata", response_model=DocumentInfo)
async def get_document_metadata(document_id: str):
    """
    Get metadata for a specific document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Document metadata
    """
    try:
        # Check if using Supabase
        use_supabase = settings.environment == "production" or settings.use_supabase_storage
        
        if use_supabase and settings.supabase_url and settings.supabase_key:
            # Get from Supabase
            try:
                from src.storage.supabase_client import get_supabase_storage
                supabase_storage = get_supabase_storage()
                
                doc = supabase_storage.get_document(document_id)
                if not doc:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                # Convert Supabase format to DocumentInfo
                return DocumentInfo(
                    document_id=doc['id'],
                    filename=doc['filename'],
                    file_type=doc.get('file_type'),
                    upload_timestamp=doc.get('upload_date'),
                    authors=doc.get('metadata', {}).get('authors'),
                    year=doc.get('metadata', {}).get('year'),
                    keywords=doc.get('metadata', {}).get('keywords'),
                    abstract=doc.get('metadata', {}).get('abstract'),
                    doi=doc.get('metadata', {}).get('doi'),
                    arxiv_id=doc.get('metadata', {}).get('arxiv_id'),
                    venue=doc.get('metadata', {}).get('venue')
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Failed to get metadata from Supabase: {e}")
                # Fall back to ChromaDB
        
        # Get from ChromaDB (local)
        vector_store = get_vector_store()
        metadata = vector_store.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(**metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}/metadata")
async def update_document_metadata(
    document_id: str,
    metadata_update: "DocumentMetadataUpdate"  # type: ignore
):
    """
    Update metadata for a document.
    
    Args:
        document_id: Document ID
        metadata_update: Metadata fields to update
        
    Returns:
        Success response with number of chunks updated
    """
    try:
        from src.api.schemas import DocumentMetadataUpdate
        
        # Convert pydantic model to dict, excluding None values
        update_dict = metadata_update.model_dump(exclude_none=True)
        
        # Check if using Supabase
        use_supabase = settings.environment == "production" or settings.use_supabase_storage
        chunks_updated = 0
        
        if use_supabase and settings.supabase_url and settings.supabase_key:
            # Update in Supabase
            try:
                from src.storage.supabase_client import get_supabase_storage
                supabase_storage = get_supabase_storage()
                
                # Get document first to check if exists
                doc = supabase_storage.get_document(document_id)
                if not doc:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                # Update document metadata in Supabase
                # Merge with existing metadata
                existing_metadata = doc.get('metadata', {})
                existing_metadata.update(update_dict)
                
                supabase_storage.update_document(document_id, {
                    'metadata': existing_metadata
                })
                
                chunks_updated = doc.get('chunk_count', 0)
                logger.info(f"✅ Updated metadata for document {document_id} in Supabase")
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "chunks_updated": chunks_updated,
                    "updated_fields": list(update_dict.keys())
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Failed to update metadata in Supabase: {e}")
                # Fall back to ChromaDB
        
        # Update in ChromaDB (local)
        vector_store = get_vector_store()
        chunks_updated = vector_store.update_document_metadata(document_id, update_dict)
        
        if chunks_updated == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Updated metadata for document {document_id}: {chunks_updated} chunks")
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunks_updated": chunks_updated,
            "updated_fields": list(update_dict.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(request: DocumentSearchRequest):
    """
    Search documents with metadata filters.
    
    Args:
        request: Search request with query and filters
        
    Returns:
        Matching documents
    """
    try:
        from src.api.schemas import DocumentSearchResponse
        
        vector_store = get_vector_store()
        
        # Perform search
        results = vector_store.search_documents(
            query=request.query,
            authors=request.authors,
            year_min=request.year_min,
            year_max=request.year_max,
            keywords=request.keywords
        )
        
        # Convert to DocumentInfo objects
        doc_infos = [DocumentInfo(**doc) for doc in results]
        
        logger.info(f"Document search returned {len(doc_infos)} results")
        
        return DocumentSearchResponse(
            documents=doc_infos,
            total=len(doc_infos),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

