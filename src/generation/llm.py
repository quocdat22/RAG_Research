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
 - ALWAYS cite sources using numbered citations [1], [2], etc. immediately after each claim or fact
 - Each context chunk has a citation number - use that exact number when citing
 - Place citations right after the specific claim they support, NOT at the end of sentences or paragraphs
 - You can use the same citation number multiple times if citing the same source
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
                f"[Citation {i}] From: {filename}, Page {page}\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    @staticmethod
    def create_user_prompt(query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        Create user prompt with query, context, and optional conversation history.
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            conversation_history: Optional list of previous messages
            
        Returns:
            Complete user prompt
        """
        history_section = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-10:]:  # Limit to last 10 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                # Truncate long messages
                content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
                history_parts.append(f"{role}: {content}")
            history_section = f"""Previous conversation:

{chr(10).join(history_parts)}

---

"""
        
        return f"""{history_section}Context from research documents:

{context}

---

Question: {query}

Please provide a comprehensive answer based on the context above. Use numbered citations [1], [2], etc. immediately after each claim."""


class OpenAIGenerator:
    """OpenAI LLM wrapper for generating RAG responses."""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for API (optional)
            model: Model name (defaults to settings.llm.model)
            temperature: Sampling temperature (defaults to settings.llm.temperature)
            max_tokens: Maximum response tokens (defaults to settings.llm.max_tokens)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or settings.llm.model
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_tokens = max_tokens or settings.llm.max_tokens
        
        logger.info(f"Initialized OpenAIGenerator with model={self.model}")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        stream: bool = False,
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Generate response based on query and retrieved chunks.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks
            stream: Whether to stream the response
            conversation_history: Optional previous conversation messages
            
        Returns:
            Generated response text
        """
        # Format context
        context = RAGPromptTemplate.format_context(retrieved_chunks)
        user_prompt = RAGPromptTemplate.create_user_prompt(query, context, conversation_history)
        
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
        Extract citation information from retrieved chunks with confidence scores.
        
        Args:
            response: Generated response
            retrieved_chunks: Retrieved chunks that were used
            
        Returns:
            List of citation metadata with confidence scores and indices
        """
        import re
        
        citations = []
        
        # Extract all citation numbers used in the response
        citation_pattern = r'\[(\d+)\]'
        used_citations = set(int(match) for match in re.findall(citation_pattern, response))
        
        # Process each chunk
        for i, chunk in enumerate(retrieved_chunks, start=1):
            # Only include citations that were actually used in the response
            # If no citations found in text, include all chunks (backward compatibility)
            if used_citations and i not in used_citations:
                continue
                
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            page = metadata.get("page_number", "?")
            
            # Calculate confidence score from retrieval score
            # Retrieval scores vary by search type:
            # - Vector: cosine similarity (0-1, higher is better)
            # - BM25: BM25 score (unbounded, higher is better)
            # - Hybrid: RRF score (0-1, higher is better)
            score = chunk.get("score", 0)
            
            # Normalize to 0-100% range
            # For most cases, we'll use a sigmoid-like transformation
            if score > 1.0:
                # BM25 scores - normalize using max observed (~30) as reference
                confidence = min(100.0, (score / 30.0) * 100.0)
            else:
                # Vector/Hybrid scores (0-1 range) - direct conversion
                confidence = score * 100.0
            
            # Apply ranking-based boost: earlier chunks get slight boost
            # Top result: +0%, 2nd: -5%, 3rd: -10%, etc (max -20%)
            rank_penalty = min(20.0, (i - 1) * 5.0)
            confidence = max(0.0, min(100.0, confidence - rank_penalty))
            
            citations.append({
                "filename": filename,
                "page": page,
                "file_type": metadata.get("file_type", "unknown"),
                "confidence_score": round(confidence, 1),
                "citation_index": i
            })
        
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
