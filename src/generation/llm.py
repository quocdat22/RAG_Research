"""OpenAI LLM wrapper for RAG generation."""
import logging
from typing import Dict, Iterator, List, Optional

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class RAGPromptTemplate:
    """Prompt templates for RAG responses."""
    
    SYSTEM_PROMPT = """You are a helpful research assistant. Your task is to answer questions based ONLY on the provided context from research documents.
 
 Guidelines:
 - Use ONLY information from the provided context
 - Always cite your sources using the format: [Source: filename, page X]
 - If the context doesn't contain enough information to answer the question, say so
 - Be concise but comprehensive
 - Maintain academic tone
 - **IMPORTANT**: For mathematical formulas, use LaTeX notation. Use single dollar signs ($) for inline formulas (e.g., $E=mc^2$) and double dollar signs ($$) for block/centered formulas (e.g., $$A = \\pi r^2$$).
 """
    
    @staticmethod
    def format_context(retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            retrieved_chunks: List of retrieved chunks with text and metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            page = metadata.get("page_number", "?")
            
            context_parts.append(
                f"[Chunk {i}] From: {filename}, Page {page}\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    @staticmethod
    def create_user_prompt(query: str, context: str) -> str:
        """
        Create user prompt with query and context.
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            Complete user prompt
        """
        return f"""Context from research documents:

{context}

---

Question: {query}

Please provide a comprehensive answer based on the context above, and cite your sources."""


class OpenAIGenerator:
    """OpenAI LLM wrapper for generating RAG responses."""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized OpenAIGenerator with model={model}")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        stream: bool = False
    ) -> str:
        """
        Generate response based on query and retrieved chunks.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        # Format context
        context = RAGPromptTemplate.format_context(retrieved_chunks)
        user_prompt = RAGPromptTemplate.create_user_prompt(query, context)
        
        # Create messages
        messages = [
            {"role": "system", "content": RAGPromptTemplate.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if stream:
                return self._generate_stream(messages)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                answer = response.choices[0].message.content
                logger.info(f"Generated response for query: '{query[:50]}...'")
                return answer
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _generate_stream(self, messages: List[Dict]) -> Iterator[str]:
        """
        Generate streaming response.
        
        Args:
            messages: Chat messages
            
        Yields:
            Response chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise
    
    def extract_citations(self, response: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Extract citation information from retrieved chunks.
        
        Args:
            response: Generated response
            retrieved_chunks: Retrieved chunks that were used
            
        Returns:
            List of citation metadata
        """
        citations = []
        seen = set()
        
        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            page = metadata.get("page_number", "?")
            
            # Create unique citation key
            citation_key = f"{filename}_{page}"
            
            if citation_key not in seen:
                citations.append({
                    "filename": filename,
                    "page": page,
                    "file_type": metadata.get("file_type", "unknown")
                })
                seen.add(citation_key)
        
        return citations


def get_generator(model_name: Optional[str] = None) -> OpenAIGenerator:
    """Get configured generator instance."""
    return OpenAIGenerator(
        api_key=settings.api_key,
        base_url=settings.api_base_url,
        model=model_name or settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens
    )
