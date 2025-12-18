"""LlamaParse PDF loader for complex document processing."""
import logging
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from src.ingestion.loaders import DocumentMetadata, DocumentPage

logger = logging.getLogger(__name__)


class LlamaParseLoader:
    """
    Load and parse PDF documents using LlamaParse for complex layouts.
    
    LlamaParse returns markdown with structured tables, preserving
    document structure for better LLM reasoning.
    """
    
    def __init__(self):
        """Initialize LlamaParse loader if available."""
        self._parser = None
        self._is_available = False
        
        if settings.llamaparse.is_available:
            try:
                from llama_parse import LlamaParse
                
                self._parser = LlamaParse(
                    api_key=settings.llama_cloud_api_key,
                    result_type=settings.llamaparse.result_type,
                    verbose=False
                )
                self._is_available = True
                logger.info("LlamaParse initialized successfully")
            except ImportError:
                logger.warning("llama-parse package not installed, falling back to pypdf")
            except Exception as e:
                logger.warning(f"Failed to initialize LlamaParse: {e}")
        else:
            if not settings.llamaparse.enabled:
                logger.info("LlamaParse is disabled in settings")
            if not settings.llama_cloud_api_key:
                logger.warning("LlamaParse API key (LLAMA_CLOUD_API_KEY) is not set")
    
    @property
    def is_available(self) -> bool:
        """Check if LlamaParse is available and configured."""
        return self._is_available
    
    def load(self, file_path: Path) -> List[DocumentPage]:
        """
        Load a PDF file using LlamaParse.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of DocumentPage objects with markdown content
        """
        if not self._is_available or not self._parser:
            raise RuntimeError("LlamaParse is not available")
        
        try:
            # Parse the document
            documents = self._parser.load_data(str(file_path))
            
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_path=str(file_path),
                file_type="pdf",
                page_count=len(documents)
            )
            
            pages = []
            for i, doc in enumerate(documents, start=1):
                # LlamaParse returns Document objects with text attribute
                content = doc.text if hasattr(doc, 'text') else str(doc)
                
                if content.strip():
                    pages.append(DocumentPage(
                        content=content,
                        page_number=i,
                        metadata=metadata
                    ))
            
            logger.info(f"LlamaParse loaded PDF: {file_path.name} ({len(pages)} pages)")
            return pages
            
        except Exception as e:
            logger.error(f"LlamaParse failed for {file_path}: {e}")
            raise


# Singleton instance
_llamaparse_loader: Optional[LlamaParseLoader] = None


def get_llamaparse_loader() -> LlamaParseLoader:
    """Get or create the LlamaParse loader singleton."""
    global _llamaparse_loader
    if _llamaparse_loader is None:
        _llamaparse_loader = LlamaParseLoader()
    return _llamaparse_loader
