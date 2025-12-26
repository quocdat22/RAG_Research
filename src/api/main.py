"""FastAPI application entry point."""
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Ensure standard streams use UTF-8 on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import chat, conversations, documents, search
from src.api.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.log_dir / "api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting RAG Native API...")
    logger.info(f"ChromaDB directory: {settings.chroma_dir}")
    logger.info(f"Documents directory: {settings.documents_dir}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Native API...")


# Create FastAPI app
app = FastAPI(
    title="RAG Native API",
    description="Research Assistant RAG System API",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(conversations.router)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    try:
        from src.storage.vector_store import get_vector_store
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            collection_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow()
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check."""
    return await root()


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Only use reload in development
    is_dev = os.getenv("ENVIRONMENT", "development") == "development"
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=is_dev
    )
