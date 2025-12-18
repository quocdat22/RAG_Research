"""Metadata extraction service for research documents."""
import logging
import re
from typing import Dict, List, Optional

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extract rich metadata from research documents using LLM and regex."""
    
    def __init__(self):
        """Initialize metadata extractor."""
        # Initialize OpenAI client for metadata extraction
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.api_base_url
        )
        logger.info("Initialized MetadataExtractor")
    
    def extract(self, document_text: str, filename: str) -> Dict[str, Optional[str]]:
        """
        Extract metadata from document text.
        
        Args:
            document_text: First few pages of the document
            filename: Original filename for context
            
        Returns:
            Dictionary with extracted metadata:
            - authors: Comma-separated list of authors
            - year: Publication year
            - keywords: Comma-separated keywords
            - abstract: Document abstract
            - doi: Digital Object Identifier
            - arxiv_id: arXiv identifier
            - venue: Conference or journal name
        """
        metadata = {}
        
        # Try LLM-based extraction first
        try:
            llm_metadata = self._extract_with_llm(document_text, filename)
            metadata.update(llm_metadata)
        except Exception as e:
            logger.warning(f"LLM metadata extraction failed: {e}")
        
        # Apply regex-based extraction to fill gaps
        regex_metadata = self._extract_with_regex(document_text)
        for key, value in regex_metadata.items():
            if not metadata.get(key):
                metadata[key] = value
        
        logger.info(f"Extracted metadata for {filename}: {list(metadata.keys())}")
        return metadata
    
    def _extract_with_llm(self, text: str, filename: str) -> Dict[str, Optional[str]]:
        """
        Use LLM to extract metadata from document text.
        
        Args:
            text: Document text (first few pages)
            filename: Filename for context
            
        Returns:
            Dictionary with extracted metadata
        """
        # Truncate text to ~4000 characters to stay within token limits
        truncated_text = text[:4000]
        
        prompt = f"""Analyze the following research document and extract metadata. Return ONLY a JSON object with these fields (use null if not found):

{{
  "authors": "comma-separated list of author names",
  "year": "publication year (4 digits)",
  "keywords": "comma-separated keywords or key terms",
  "abstract": "document abstract or summary",
  "doi": "DOI identifier if present",
  "arxiv_id": "arXiv ID if present",
  "venue": "conference or journal name"
}}

Filename: {filename}

Document text:
{truncated_text}

JSON output:"""
        
        try:
            # Use lightweight model for metadata extraction
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a metadata extraction assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse JSON from response
            import json
            # Remove markdown code blocks if present
            if result.startswith("```"):
                result = re.sub(r'^```(?:json)?\s*', '', result)
                result = re.sub(r'\s*```$', '', result)
            
            metadata = json.loads(result)
            
            # Convert null to None and clean up
            for key in metadata:
                if metadata[key] == "null" or metadata[key] is None:
                    metadata[key] = None
                elif isinstance(metadata[key], str):
                    metadata[key] = metadata[key].strip()
            
            return metadata
            
        except Exception as e:
            logger.error(f"LLM metadata extraction error: {e}")
            return {}
    
    def _extract_with_regex(self, text: str) -> Dict[str, Optional[str]]:
        """
        Use regex patterns to extract metadata.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract DOI
        doi_pattern = r'(?:doi:|DOI:)?\s*(10\.\d{4,}(?:\.\d+)*\/\S+)'
        doi_match = re.search(doi_pattern, text, re.IGNORECASE)
        if doi_match:
            metadata['doi'] = doi_match.group(1).rstrip('.,;')
        
        # Extract arXiv ID
        arxiv_pattern = r'arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)'
        arxiv_match = re.search(arxiv_pattern, text, re.IGNORECASE)
        if arxiv_match:
            metadata['arxiv_id'] = arxiv_match.group(1)
        
        # Extract year (4 digits)
        year_pattern = r'\b(19|20)\d{2}\b'
        year_matches = re.findall(year_pattern, text)
        if year_matches:
            # Take the first reasonable year found
            years = [y for y in year_matches if 1990 <= int(y) <= 2030]
            if years:
                metadata['year'] = years[0]
        
        return metadata


# Singleton instance
_metadata_extractor = None


def get_metadata_extractor() -> MetadataExtractor:
    """Get or create metadata extractor instance."""
    global _metadata_extractor
    if _metadata_extractor is None:
        _metadata_extractor = MetadataExtractor()
    return _metadata_extractor
