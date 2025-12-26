"""Context resolver for handling coreference resolution in conversations."""
import logging
from typing import Dict, List, Optional

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class ContextResolver:
    """
    Resolves coreferences in user queries using conversation history.
    
    Handles pronouns like "it", "this", "that" (Vietnamese: "nó", "điều này", "bài báo đó")
    and expands them to their full references based on conversation context.
    """
    
    REWRITE_PROMPT = """You are a query rewriter. Your task is to rewrite the user's current query to be self-contained by resolving any pronouns or references using the conversation history.

Rules:
1. If the query contains pronouns or references (like "it", "this", "that", "they", "nó", "điều này", "bài báo đó"), replace them with the actual entities they refer to from the conversation history.
2. If the query is already self-contained, return it unchanged.
3. Keep the rewritten query concise and natural.
4. Return ONLY the rewritten query, nothing else.

Conversation history:
{history}

Current query: {query}

Rewritten query:"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize context resolver.
        
        Args:
            api_key: API key for LLM
            base_url: Base URL for API
            model: Model to use for query rewriting (defaults to settings.llm.model)
        """
        self.client = OpenAI(
            api_key=api_key or settings.api_key,
            base_url=base_url or settings.api_base_url
        )
        self.model = model or settings.llm.model
        logger.info(f"ContextResolver initialized with model={self.model}")
    
    def _format_history(self, messages: List[Dict]) -> str:
        """
        Format conversation history for the prompt.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted history string
        """
        if not messages:
            return "No previous conversation."
        
        history_parts = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long messages to save tokens
            content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def _needs_resolution(self, query: str) -> bool:
        """
        Quick check if query might need coreference resolution.
        
        Args:
            query: User's query
            
        Returns:
            True if query might contain references needing resolution
        """
        # Common pronouns and references in English and Vietnamese
        reference_words = [
            # English
            "it", "this", "that", "they", "them", "these", "those",
            "the paper", "the article", "the document", "the above",
            # Vietnamese
            "nó", "điều này", "điều đó", "bài báo đó", "tài liệu đó",
            "cái này", "cái đó", "những cái đó", "ở trên"
        ]
        
        query_lower = query.lower()
        return any(ref in query_lower for ref in reference_words)
    
    def resolve(
        self,
        query: str,
        conversation_history: List[Dict],
        force_resolve: bool = False
    ) -> str:
        """
        Resolve coreferences in the query using conversation history.
        
        Args:
            query: User's current query
            conversation_history: List of previous messages with 'role' and 'content'
            force_resolve: If True, always attempt resolution even if no obvious references
            
        Returns:
            Resolved query (may be same as input if no resolution needed)
        """
        # Skip if no history or query doesn't seem to need resolution
        if not conversation_history:
            return query
        
        if not force_resolve and not self._needs_resolution(query):
            return query
        
        try:
            # Format history for prompt
            history_text = self._format_history(conversation_history[-10:])  # Last 10 messages
            
            prompt = self.REWRITE_PROMPT.format(
                history=history_text,
                query=query
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            
            # Validate the rewritten query
            if not rewritten_query or len(rewritten_query) < 3:
                logger.warning("Invalid rewritten query, using original")
                return query
            
            if rewritten_query != query:
                logger.info(f"Query resolved: '{query}' -> '{rewritten_query}'")
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error resolving coreferences: {e}")
            return query  # Fall back to original query


# Global resolver instance
_context_resolver: Optional[ContextResolver] = None


def get_context_resolver() -> ContextResolver:
    """Get or create global context resolver instance."""
    global _context_resolver
    if _context_resolver is None:
        _context_resolver = ContextResolver()
    return _context_resolver
