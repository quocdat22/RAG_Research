"""OpenAI embeddings wrapper with batching and retry logic."""
import logging
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """Wrapper for OpenAI embeddings API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for API (optional)
            model: Embedding model name (defaults to settings.embedding.model)
            batch_size: Number of texts to embed in one API call
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or settings.embedding.model
        self.batch_size = batch_size
        
        logger.info(f"Initialized OpenAIEmbedder with model={self.model}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Embedded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts with batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        
        logger.info(f"Embedded {len(texts)} texts in {len(texts) // self.batch_size + 1} batches")
        return all_embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self._embed_batch([text])[0]


def get_embedder() -> OpenAIEmbedder:
    """Get configured embedder instance."""
    return OpenAIEmbedder(
        api_key=settings.api_key,
        base_url=settings.api_base_url,
        model=settings.embedding.model,
        batch_size=100
    )
