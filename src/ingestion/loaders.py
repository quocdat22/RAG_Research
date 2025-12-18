"""Document loaders for PDF, DOCX, and TXT files."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pypdf
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Metadata for a loaded document."""
    
    def __init__(
        self,
        filename: str,
        file_path: str,
        file_type: str,
        upload_timestamp: Optional[datetime] = None,
        page_count: Optional[int] = None,
        # Rich metadata fields
        authors: Optional[str] = None,
        year: Optional[str] = None,
        keywords: Optional[str] = None,
        abstract: Optional[str] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        venue: Optional[str] = None
    ):
        self.filename = filename
        self.file_path = file_path
        self.file_type = file_type
        self.upload_timestamp = upload_timestamp or datetime.utcnow()
        self.page_count = page_count
        
        # Rich metadata
        self.authors = authors
        self.year = year
        self.keywords = keywords
        self.abstract = abstract
        self.doi = doi
        self.arxiv_id = arxiv_id
        self.venue = venue
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "page_count": self.page_count,
            # Rich metadata
            "authors": self.authors,
            "year": self.year,
            "keywords": self.keywords,
            "abstract": self.abstract,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "venue": self.venue
        }


class DocumentPage:
    """Represents a page or section of a document."""
    
    def __init__(
        self,
        content: str,
        page_number: Optional[int] = None,
        metadata: Optional[DocumentMetadata] = None
    ):
        self.content = content
        self.page_number = page_number
        self.metadata = metadata or DocumentMetadata("", "", "")
    
    def to_dict(self) -> Dict:
        """Convert page to dictionary."""
        return {
            "content": self.content,
            "page_number": self.page_number,
            "metadata": self.metadata.to_dict()
        }


class PDFLoader:
    """Load and parse PDF documents."""
    
    @staticmethod
    def load(file_path: Path) -> List[DocumentPage]:
        """
        Load a PDF file and extract text with page numbers.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of DocumentPage objects
        """
        try:
            pages = []
            with open(file_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                page_count = len(pdf_reader.pages)
                
                metadata = DocumentMetadata(
                    filename=file_path.name,
                    file_path=str(file_path),
                    file_type="pdf",
                    page_count=page_count
                )
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():  # Only include pages with content
                        pages.append(DocumentPage(
                            content=text,
                            page_number=page_num,
                            metadata=metadata
                        ))
                
                logger.info(f"Loaded PDF: {file_path.name} ({page_count} pages)")
                return pages
                
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise


class DOCXLoader:
    """Load and parse DOCX documents."""
    
    @staticmethod
    def load(file_path: Path) -> List[DocumentPage]:
        """
        Load a DOCX file and extract text.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List of DocumentPage objects (one per paragraph group)
        """
        try:
            doc = DocxDocument(file_path)
            
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_path=str(file_path),
                file_type="docx"
            )
            
            # Group paragraphs into pages (simulate pages with ~500 chars each)
            pages = []
            current_page_content = []
            current_length = 0
            page_num = 1
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                current_page_content.append(text)
                current_length += len(text)
                
                # Create a "page" when we reach ~2000 characters
                if current_length >= 2000:
                    pages.append(DocumentPage(
                        content="\n".join(current_page_content),
                        page_number=page_num,
                        metadata=metadata
                    ))
                    current_page_content = []
                    current_length = 0
                    page_num += 1
            
            # Add remaining content as last page
            if current_page_content:
                pages.append(DocumentPage(
                    content="\n".join(current_page_content),
                    page_number=page_num,
                    metadata=metadata
                ))
            
            logger.info(f"Loaded DOCX: {file_path.name} ({len(pages)} sections)")
            return pages
            
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise


class TXTLoader:
    """Load and parse TXT documents."""
    
    @staticmethod
    def load(file_path: Path) -> List[DocumentPage]:
        """
        Load a TXT file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            List of DocumentPage objects
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            metadata = DocumentMetadata(
                filename=file_path.name,
                file_path=str(file_path),
                file_type="txt"
            )
            
            # For TXT, we'll create one page
            pages = [DocumentPage(
                content=content,
                page_number=1,
                metadata=metadata
            )]
            
            logger.info(f"Loaded TXT: {file_path.name}")
            return pages
            
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise


class DocumentLoader:
    """Main document loader with automatic type detection."""
    
    LOADERS = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
        ".txt": TXTLoader
    }
    
    @classmethod
    def load(cls, file_path: Path, use_llamaparse: bool = True) -> tuple[List[DocumentPage], bool]:
        """
        Load a document based on its file extension.
        
        Args:
            file_path: Path to document file
            use_llamaparse: Whether to try LlamaParse for PDFs
            
        Returns:
            Tuple of (List of DocumentPage objects, is_markdown flag)
            is_markdown is True when LlamaParse was used
            
        Raises:
            ValueError: If file type is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix not in cls.LOADERS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: {', '.join(cls.LOADERS.keys())}"
            )
        
        # Try LlamaParse for PDFs if enabled
        if suffix == ".pdf" and use_llamaparse:
            try:
                from src.ingestion.llama_parser import get_llamaparse_loader
                
                llamaparse = get_llamaparse_loader()
                if llamaparse.is_available:
                    pages = llamaparse.load(file_path)
                    return pages, True  # is_markdown = True
            except Exception as e:
                logger.warning(f"LlamaParse failed, falling back to pypdf: {e}")
        
        # Fallback to standard loader
        loader_class = cls.LOADERS[suffix]
        return loader_class.load(file_path), False  # is_markdown = False

