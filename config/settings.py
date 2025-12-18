"""Application settings using Pydantic."""
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Embedding configuration."""
    
    model: str = Field(default="openai/text-embedding-3-small", description="Embedding model")
    dimension: int = Field(default=1536, description="Embedding dimension")
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class LLMSettings(BaseSettings):
    """LLM configuration."""
    
    model: str = Field(default="openai/gpt-4.1-mini", description="Chat model")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, le=16000, description="Max response tokens")
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class ChunkingSettings(BaseSettings):
    """Chunking configuration."""
    
    size: int = Field(default=800, ge=500, le=1000, description="Chunk size in tokens")
    overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks in tokens")
    
    model_config = SettingsConfigDict(env_prefix="CHUNK_")
    
    @field_validator("size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is within recommended range."""
        if not 500 <= v <= 1000:
            raise ValueError("Chunk size must be between 500 and 1000 tokens")
        return v
    
    @field_validator("overlap")
    @classmethod
    def validate_overlap(cls, v: int) -> int:
        """Validate overlap is reasonable."""
        if not 0 <= v <= 500:
            raise ValueError("Chunk overlap must be between 0 and 500 tokens")
        return v


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""
    
    top_k: int = Field(default=3, ge=1, le=20, description="Number of results to return after all steps")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector search")
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for BM25 search")
    
    model_config = SettingsConfigDict(env_prefix="")
    
    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k is within recommended range."""
        if not 1 <= v <= 20:
            raise ValueError("top_k must be between 1 and 20")
        return v


class RerankSettings(BaseSettings):
    """Reranking configuration."""
    
    enabled: bool = Field(default=True, description="Whether to enable reranking")
    model: str = Field(default="rerank-v4.0-fast", description="Cohere rerank model")
    top_n: int = Field(default=3, ge=1, le=10, description="Number of chunks to keep after reranking")
    initial_top_k: int = Field(default=5, ge=3, le=20, description="Number of chunks to retrieve before reranking")
    
    model_config = SettingsConfigDict(env_prefix="RERANK_")


class LlamaParseSettings(BaseSettings):
    """LlamaParse configuration for complex PDF parsing."""
    
    enabled: bool = Field(default=True, description="Whether to enable LlamaParse for PDF processing", validation_alias="LLAMAPARSE_ENABLED")
    result_type: str = Field(default="markdown", validation_alias="LLAMAPARSE_RESULT_TYPE")
    output_tables_as_html: bool = Field(default=False, validation_alias="LLAMAPARSE_OUTPUT_TABLES_AS_HTML")
    
    model_config = SettingsConfigDict(populate_by_name=True)
    
    @property
    def is_available(self) -> bool:
        """Check if LlamaParse is enabled and has API key."""
        # Use the key from the main settings if available
        from config.settings import settings
        return self.enabled and bool(settings.llama_cloud_api_key)


class Settings(BaseSettings):
    """Main application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    github_token: Optional[str] = Field(default=None, description="GitHub token for GitHub Models API")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key for reranking")
    llama_cloud_api_key: Optional[str] = Field(default=None, description="LlamaCloud API key", validation_alias="LLAMA_CLOUD_API_KEY")
    
    # API Endpoints
    api_base_url: Optional[str] = Field(
        default="https://models.github.ai/inference", 
        description="Base URL for the API (OpenAI or GitHub Models)"
    )
    
    @property
    def api_key(self) -> str:
        """Get the available API key (prioritizes GitHub token)."""
        if self.github_token:
            return self.github_token
        if self.openai_api_key:
            return self.openai_api_key
        raise ValueError("No API key found. Please set GITHUB_TOKEN or OPENAI_API_KEY.")
    
    # Sub-settings
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    llamaparse: LlamaParseSettings = Field(default_factory=LlamaParseSettings)
    
    # Storage paths
    documents_dir: Path = Field(default=Path("./data/documents"), description="Documents storage directory")
    chroma_dir: Path = Field(default=Path("./data/chroma_db"), description="ChromaDB storage directory")
    log_dir: Path = Field(default=Path("./logs"), description="Logs directory")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def model_post_init(self, __context) -> None:
        """Create necessary directories after model initialization."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
