"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    filename: str
    chunk_count: int
    status: str = "success"


class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str
    filename: str
    file_type: str
    upload_timestamp: str
    # Rich metadata fields
    authors: Optional[str] = None
    year: Optional[str] = None
    keywords: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: List[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: str
    chunks_deleted: int
    status: str = "success"


class ChunkInfo(BaseModel):
    """Information about a single text chunk."""
    chunk_id: str
    text: str
    metadata: Dict


class DocumentChunksResponse(BaseModel):
    """Response containing all chunks for a document."""
    document_id: str
    chunks: List[ChunkInfo]
    total: int


class SearchRequest(BaseModel):
    """Request for document search."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    search_type: Literal["vector", "bm25", "hybrid"] = Field("hybrid", description="Search method")


class SearchResult(BaseModel):
    """Individual search result."""
    text: str
    score: float
    metadata: Dict


class SearchResponse(BaseModel):
    """Response for search request."""
    query: str
    results: List[SearchResult]
    search_type: str


class SourceCitation(BaseModel):
    """Source citation information."""
    filename: str
    page: int | str
    file_type: str
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence score (0-100%)")
    citation_index: int = Field(..., description="Citation number in the response")


class ChatRequest(BaseModel):
    """Request for chat/Q&A."""
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(3, ge=1, le=20, description="Number of chunks to retrieve")
    search_type: Literal["vector", "bm25", "hybrid"] = Field("hybrid", description="Search method")
    model_mode: Literal["light", "full"] = Field("light", description="LLM model mode")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context continuity")


class ChatResponse(BaseModel):
    """Response for chat/Q&A."""
    query: str
    answer: str
    sources: List[SourceCitation]
    search_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    collection_stats: Optional[Dict] = None


# Conversation schemas
class MessageSchema(BaseModel):
    """Schema for a conversation message."""
    id: str
    role: Literal["user", "assistant"]
    content: str
    sources: Optional[List[SourceCitation]] = None
    created_at: datetime


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    title: Optional[str] = Field(None, description="Optional title for the conversation")


class ConversationUpdate(BaseModel):
    """Request to update a conversation."""
    title: str = Field(..., min_length=1, description="New title for the conversation")


class ConversationResponse(BaseModel):
    """Response containing a single conversation with messages."""
    id: str
    title: str
    messages: List[MessageSchema] = []
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response containing list of conversations."""
    conversations: List[ConversationResponse]
    total: int


# Document metadata schemas
class DocumentMetadataUpdate(BaseModel):
    """Request to update document metadata."""
    authors: Optional[str] = Field(None, description="Comma-separated author names")
    year: Optional[str] = Field(None, description="Publication year")
    keywords: Optional[str] = Field(None, description="Comma-separated keywords")
    abstract: Optional[str] = Field(None, description="Document abstract")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")
    venue: Optional[str] = Field(None, description="Conference or journal name")


class DocumentSearchRequest(BaseModel):
    """Request for smart document search."""
    query: Optional[str] = Field(None, description="Search query across all text fields")
    authors: Optional[str] = Field(None, description="Filter by authors (partial match)")
    year_min: Optional[int] = Field(None, description="Minimum publication year")
    year_max: Optional[int] = Field(None, description="Maximum publication year")
    keywords: Optional[str] = Field(None, description="Filter by keywords (partial match)")


class DocumentSearchResponse(BaseModel):
    """Response for document search."""
    documents: List[DocumentInfo]
    total: int
    query: Optional[str] = None

