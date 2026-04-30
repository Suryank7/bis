"""
Living Documentation Copilot - RAG Question Answerer

The core RAG pipeline that:
1. Receives a question
2. Retrieves relevant chunks from the vector index
3. Constructs a prompt with context
4. Generates an answer using the LLM
5. Returns the answer with source attribution

This module integrates with Pathway's streaming updates.
"""

from typing import List, Dict, Optional, Tuple
import logging

from ..indexing.embeddings import compute_embedding
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a helpful documentation assistant. Your role is to answer questions based ONLY on the provided documentation context.

Rules:
1. Answer questions accurately based on the provided context
2. If the context doesn't contain relevant information, say "I don't have information about that in the current documentation"
3. Be concise but complete
4. When citing information, mention the source document when possible
5. Never make up information that isn't in the context

Remember: You can ONLY use information from the provided context."""


RAG_USER_PROMPT_TEMPLATE = """Context from documentation:
{context}

---

Question: {question}

Please answer based on the documentation context above."""


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


class RAGQuestionAnswerer:
    """
    Real-time RAG Question Answerer.
    
    This class provides the main interface for asking questions
    against the live documentation index.
    
    Features:
    - Semantic search using embeddings
    - Context-aware answer generation
    - Source attribution
    - Works with streaming document updates
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        temperature: float = 0.7
    ):
        """
        Initialize the RAG Question Answerer.
        
        Args:
            llm_client: Configured LLM client
            embedding_model: Model for query embedding
            top_k: Number of chunks to retrieve
            temperature: LLM generation temperature
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.temperature = temperature
        
        # Storage for indexed chunks (updated by pipeline)
        self._chunks_data: List[Dict] = []
        
        logger.info(f"🔄 Initializing RAG Question Answerer")
        logger.info(f"   LLM: {llm_client.client_type}")
        logger.info(f"   Top-K: {top_k}")
    
    def update_index(self, chunks: List[Dict]):
        """
        Update the in-memory index with new chunk data.
        
        Called by the pipeline when documents change.
        
        Args:
            chunks: List of chunk dicts with text, embedding, and metadata
        """
        self._chunks_data = chunks
        logger.info(f"📊 Index updated: {len(chunks)} chunks")
    
    def _retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: The user's question
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self._chunks_data:
            return []
        
        # Compute query embedding
        query_embedding = compute_embedding(query, self.embedding_model)
        
        # Score all chunks
        scored_chunks = []
        for chunk in self._chunks_data:
            chunk_embedding = chunk.get("embedding", [])
            if chunk_embedding:
                score = cosine_similarity(query_embedding, chunk_embedding)
                scored_chunks.append({
                    **chunk,
                    "similarity_score": score
                })
        
        # Sort by similarity and take top-k
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return scored_chunks[:self.top_k]
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context string."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_attribution", chunk.get("filename", "Unknown"))
            text = chunk.get("chunk_text", "")
            
            context_parts.append(f"[Source: {source}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_sources(self, chunks: List[Dict]) -> List[str]:
        """Extract unique source documents from retrieved chunks."""
        sources = []
        seen = set()
        
        for chunk in chunks:
            doc_id = chunk.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                sources.append({
                    "document_id": doc_id,
                    "filename": chunk.get("filename", ""),
                    "source_path": chunk.get("source_path", ""),
                })
        
        return sources
    
    def answer(self, question: str) -> Dict:
        """
        Answer a question based on the current documentation.
        
        This is the main entry point for RAG queries.
        
        Args:
            question: The user's natural language question
            
        Returns:
            Dict containing:
            - answer: The generated answer
            - sources: List of source documents used
            - num_chunks_used: Number of chunks in context
            - has_context: Whether relevant context was found
        """
        logger.info(f"❓ Processing question: {question[:100]}...")
        
        # Retrieve relevant chunks
        relevant_chunks = self._retrieve(question)
        
        if not relevant_chunks:
            return {
                "answer": "I don't have any documentation indexed yet. Please add some documents to the data/docs folder.",
                "sources": [],
                "num_chunks_used": 0,
                "has_context": False
            }
        
        # Check if we have good matches
        top_score = relevant_chunks[0].get("similarity_score", 0)
        if top_score < 0.3:
            # Low confidence match
            logger.info(f"   Low confidence match (score: {top_score:.2f})")
        
        # Format context
        context = self._format_context(relevant_chunks)
        
        # Build prompt
        prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Generate answer
        logger.info(f"   Generating answer with {len(relevant_chunks)} chunks...")
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            temperature=self.temperature
        )
        
        # Get sources
        sources = self._get_sources(relevant_chunks)
        
        logger.info(f"✅ Answer generated ({len(answer)} chars)")
        
        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_used": len(relevant_chunks),
            "has_context": True,
            "top_similarity_score": top_score
        }


# Export public API
__all__ = [
    "RAGQuestionAnswerer",
    "RAG_SYSTEM_PROMPT",
]
