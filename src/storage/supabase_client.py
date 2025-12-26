"""Supabase client for storage and database operations."""
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import json

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None


class SupabaseStorage:
    """Supabase storage manager for documents and embeddings."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client.
        
        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            key: Supabase API key (defaults to SUPABASE_KEY env var)
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase-py is not installed. Install it with: uv pip install supabase")
        
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client: Client = create_client(self.url, self.key)
    
    # ========== Document Operations ==========
    
    def upload_document(
        self,
        file_path: str,
        file_content: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload document to Supabase Storage and create metadata record.
        
        Args:
            file_path: Path where file should be stored in bucket
            file_content: File content as bytes
            metadata: Additional metadata for the document
            
        Returns:
            Document record with ID and metadata
        """
        # Upload to storage with upsert option
        try:
            self.client.storage.from_("documents").upload(
                file_path,
                file_content,
                file_options={
                    "content-type": self._get_content_type(file_path),
                    "upsert": "true"  # Overwrite if exists
                }
            )
        except Exception as e:
            # If duplicate, try to remove and re-upload
            error_str = str(e)
            if "409" in error_str or "Duplicate" in error_str or "already exists" in error_str:
                try:
                    self.client.storage.from_("documents").remove([file_path])
                    self.client.storage.from_("documents").upload(
                        file_path,
                        file_content,
                        file_options={"content-type": self._get_content_type(file_path)}
                    )
                except Exception as retry_error:
                    raise Exception(f"Failed to upload after retry: {retry_error}")
            else:
                raise
        
        # Create metadata record
        doc_data = {
            "filename": Path(file_path).name,
            "file_path": file_path,
            "file_size": len(file_content),
            "file_type": Path(file_path).suffix,
            "metadata": metadata or {},
            "processed": False,
            "chunk_count": 0
        }
        
        result = self.client.table("documents").insert(doc_data).execute()
        return result.data[0] if result.data else {}
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Document metadata or None
        """
        result = self.client.table("documents").select("*").eq("id", document_id).execute()
        return result.data[0] if result.data else None
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        processed: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """List documents with optional filtering.
        
        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            processed: Filter by processed status
            
        Returns:
            List of document records
        """
        query = self.client.table("documents").select("*")
        
        if processed is not None:
            query = query.eq("processed", processed)
        
        result = query.order("upload_date", desc=True).limit(limit).offset(offset).execute()
        return result.data
    
    def update_document(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update document metadata.
        
        Args:
            document_id: UUID of the document
            updates: Fields to update
            
        Returns:
            Updated document record
        """
        result = self.client.table("documents").update(updates).eq("id", document_id).execute()
        return result.data[0] if result.data else {}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            True if successful
        """
        # Get document to get file path
        doc = self.get_document(document_id)
        if not doc:
            return False
        
        # Delete from storage
        self.client.storage.from_("documents").remove([doc["file_path"]])
        
        # Delete metadata (chunks will be cascade deleted)
        self.client.table("documents").delete().eq("id", document_id).execute()
        return True
    
    def download_document(self, file_path: str) -> bytes:
        """Download document content from storage.
        
        Args:
            file_path: Path of file in bucket
            
        Returns:
            File content as bytes
        """
        return self.client.storage.from_("documents").download(file_path)
    
    # ========== Chunk Operations ==========
    
    def save_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Save document chunks to database.
        
        Args:
            document_id: UUID of the parent document
            chunks: List of chunk data with content, embedding_id, metadata
            
        Returns:
            List of created chunk records
        """
        chunk_data = [
            {
                "document_id": document_id,
                "chunk_index": i,
                "content": chunk["content"],
                "embedding_id": chunk.get("embedding_id"),
                "metadata": chunk.get("metadata", {})
            }
            for i, chunk in enumerate(chunks)
        ]
        
        result = self.client.table("document_chunks").insert(chunk_data).execute()
        
        # Update document chunk count
        self.update_document(document_id, {
            "chunk_count": len(chunks),
            "processed": True
        })
        
        return result.data
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            List of chunk records
        """
        result = (
            self.client.table("document_chunks")
            .select("*")
            .eq("document_id", document_id)
            .order("chunk_index")
            .execute()
        )
        return result.data
    
    # ========== Conversation Operations ==========
    
    def create_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation.
        
        Args:
            title: Optional conversation title
            
        Returns:
            Conversation record
        """
        data = {"title": title} if title else {}
        result = self.client.table("conversations").insert(data).execute()
        return result.data[0] if result.data else {}
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            Conversation record or None
        """
        result = self.client.table("conversations").select("*").eq("id", conversation_id).execute()
        return result.data[0] if result.data else None
    
    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent conversations.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of conversation records
        """
        result = (
            self.client.table("conversations")
            .select("*")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Save a message to a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            role: 'user' or 'assistant'
            content: Message content
            sources: Optional list of source documents
            
        Returns:
            Message record
        """
        data = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources
        }
        result = self.client.table("messages").insert(data).execute()
        return result.data[0] if result.data else {}
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages in a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            List of message records
        """
        result = (
            self.client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        return result.data
    
    # ========== Statistics ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        result = self.client.rpc("document_stats").execute()
        return result.data[0] if result.data else {}
    
    # ========== Helpers ==========
    
    def _get_content_type(self, file_path: str) -> str:
        """Get content type from file extension."""
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json"
        }
        return content_types.get(ext, "application/octet-stream")


# Singleton instance
_supabase_storage: Optional[SupabaseStorage] = None


def get_supabase_storage() -> SupabaseStorage:
    """Get or create Supabase storage instance."""
    global _supabase_storage
    if _supabase_storage is None:
        _supabase_storage = SupabaseStorage()
    return _supabase_storage
