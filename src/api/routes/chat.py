"""Chat/Q&A routes."""
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, SourceCitation
from src.embedding.embedder import get_embedder
from src.generation.llm import get_generator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.reranker import CohereReranker
from src.storage.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _initialize_retrievers():
    """Initialize retrievers for chat."""
    vector_store = get_vector_store()
    embedder = get_embedder()
    
    # Vector retriever
    vector_retriever = VectorRetriever(vector_store, embedder)
    
    # BM25 retriever
    bm25_retriever = BM25Retriever()
    all_results = vector_store.collection.get()
    if all_results["documents"]:
        documents = [
            {"text": text, "metadata": metadata}
            for text, metadata in zip(all_results["documents"], all_results["metadatas"])
        ]
        bm25_retriever.index_documents(documents)
    
    # Hybrid retriever
    from config.settings import settings
    hybrid_retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        vector_weight=settings.retrieval.vector_weight,
        bm25_weight=settings.retrieval.bm25_weight
    )
    
    # Reranker
    reranker = None
    if settings.rerank.enabled:
        reranker = CohereReranker(
            model=settings.rerank.model,
            top_n=settings.rerank.top_n
        )
    
    return vector_retriever, bm25_retriever, hybrid_retriever, reranker


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using RAG.
    
    Args:
        request: Chat request with question
        
    Returns:
        Generated answer with source citations
    """
    try:
        # Initialize components
        vector_retriever, bm25_retriever, hybrid_retriever, reranker = _initialize_retrievers()
        # Determine model
        model_name = "openai/gpt-4.1-mini" if request.model_mode == "light" else "openai/gpt-5-chat"
        generator = get_generator(model_name=model_name)
        
        # Determine retrieval top_k
        from config.settings import settings
        retrieval_k = request.top_k
        if reranker and settings.rerank.enabled:
            # If reranking, retrieve more initially
            retrieval_k = max(request.top_k, settings.rerank.initial_top_k)
        
        # Retrieve relevant chunks
        if request.search_type == "vector":
            retrieved_chunks = vector_retriever.retrieve(request.query, top_k=retrieval_k)
        elif request.search_type == "bm25":
            retrieved_chunks = bm25_retriever.retrieve(request.query, top_k=retrieval_k)
        else:  # hybrid
            retrieved_chunks = hybrid_retriever.retrieve(request.query, top_k=retrieval_k)
            
        # Apply reranking if enabled
        if reranker and settings.rerank.enabled and retrieved_chunks:
            logger.info(f"Applying reranking to {len(retrieved_chunks)} chunks")
            retrieved_chunks = reranker.rerank(request.query, retrieved_chunks)
            # Ensure we respect the requested top_k from rerank results
            retrieved_chunks = retrieved_chunks[:request.top_k]
        
        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload documents first."
            )
        
        # Generate answer
        answer = generator.generate(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            stream=False
        )
        
        # Extract citations
        citations = generator.extract_citations(answer, retrieved_chunks)
        
        source_citations = [
            SourceCitation(
                filename=citation["filename"],
                page=citation["page"],
                file_type=citation["file_type"]
            )
            for citation in citations
        ]
        
        logger.info(f"Chat response generated for query: '{request.query[:50]}...'")
        
        return ChatResponse(
            query=request.query,
            answer=answer,
            sources=source_citations,
            search_type=request.search_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response.
    
    Args:
        request: Chat request with question
        
    Returns:
        Streaming response
    """
    try:
        # Initialize components
        vector_retriever, bm25_retriever, hybrid_retriever, reranker = _initialize_retrievers()
        # Determine model
        model_name = "openai/gpt-4.1-mini" if request.model_mode == "light" else "openai/gpt-5-chat"
        generator = get_generator(model_name=model_name)
        
        # Determine retrieval top_k
        from config.settings import settings
        retrieval_k = request.top_k
        if reranker and settings.rerank.enabled:
            # If reranking, retrieve more initially
            retrieval_k = max(request.top_k, settings.rerank.initial_top_k)
            
        # Retrieve relevant chunks
        if request.search_type == "vector":
            retrieved_chunks = vector_retriever.retrieve(request.query, top_k=retrieval_k)
        elif request.search_type == "bm25":
            retrieved_chunks = bm25_retriever.retrieve(request.query, top_k=retrieval_k)
        else:  # hybrid
            retrieved_chunks = hybrid_retriever.retrieve(request.query, top_k=retrieval_k)
            
        # Apply reranking if enabled
        if reranker and settings.rerank.enabled and retrieved_chunks:
            logger.info(f"Applying reranking (stream) to {len(retrieved_chunks)} chunks")
            retrieved_chunks = reranker.rerank(request.query, retrieved_chunks)
            # Ensure we respect the requested top_k from rerank results
            retrieved_chunks = retrieved_chunks[:request.top_k]
        
        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload documents first."
            )
        
        # Generate streaming response
        def generate_stream():
            for chunk in generator.generate(
                query=request.query,
                retrieved_chunks=retrieved_chunks,
                stream=True
            ):
                yield chunk
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
