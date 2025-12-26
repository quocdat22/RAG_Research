"""Reranker using Cohere API."""
import logging
from typing import Dict, List, Optional

try:
    import cohere
except ImportError:
    cohere = None

from config.settings import settings

logger = logging.getLogger(__name__)


class CohereReranker:
    """Reranker using Cohere's Rerank API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key (optional, defaults to settings.cohere_api_key)
            model: Rerank model name (defaults to settings.rerank.model)
            top_n: Number of chunks to keep after reranking (defaults to settings.rerank.top_n)
        """
        self.api_key = api_key or settings.cohere_api_key
        self.model = model or settings.rerank.model
        self.top_n = top_n if top_n is not None else settings.rerank.top_n
        self.client = None
        
        if cohere is None:
            logger.warning("Cohere package not installed. Reranking will be disabled.")
            return
            
        if not self.api_key:
            logger.warning("Cohere API key not found. Reranking will be disabled.")
            return
            
        try:
            self.client = cohere.ClientV2(api_key=self.api_key)
            logger.info(f"Initialized CohereReranker with model={model}")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents based on query.
        
        Args:
            query: User search query
            documents: List of retrieved chunks
            
        Returns:
            Reranked and filtered list of chunks
        """
        if not self.client or not documents:
            return documents[:self.top_n]
            
        try:
            # Prepare documents for Cohere (list of strings)
            doc_texts = [doc["text"] for doc in documents]
            
            # Call Cohere Rerank API
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=self.top_n
            )
            
            reranked_results = []
            for result in response.results:
                original_doc = documents[result.index]
                # Update score with rerank score
                reranked_results.append({
                    "text": original_doc["text"],
                    "metadata": original_doc.get("metadata", {}),
                    "score": result.relevance_score
                })
                
            logger.info(f"Successfully reranked {len(documents)} docs to top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during Cohere reranking: {e}")
            # Fallback to original order
            return documents[:self.top_n]
