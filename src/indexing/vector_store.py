"""
Living Documentation Copilot - Incremental Vector Store

This module creates a real-time, incremental vector index using Pathway.
When documents are added, modified, or deleted, the index updates automatically.

Key Features:
- Incremental updates (no full re-indexing)
- Automatic handling of document changes
- KNN similarity search
- Source attribution for retrieved chunks
"""

import pathway as pw
from typing import List, Tuple, Optional
import logging

from .embeddings import LocalEmbedder, create_embedding_udf

logger = logging.getLogger(__name__)


class IncrementalVectorIndex:
    """
    A real-time, incrementally-updating vector index.
    
    This class wraps Pathway's vector index capabilities to provide:
    - Automatic updates when source documents change
    - Efficient similarity search
    - Source tracking for retrieved chunks
    
    Usage:
        index = IncrementalVectorIndex(enriched_chunks, embedder)
        results = index.search("What is the vacation policy?", k=5)
    """
    
    def __init__(
        self,
        chunks_table: pw.Table,
        embedding_model: str = "all-MiniLM-L6-v2",
        text_column: str = "chunk_text",
    ):
        """
        Initialize the vector index.
        
        Args:
            chunks_table: Pathway table with chunk_text and metadata columns
            embedding_model: Name of sentence-transformers model to use
            text_column: Name of the column containing text to embed
        """
        self.embedding_model = embedding_model
        self.text_column = text_column
        self.embedder = LocalEmbedder(embedding_model)
        
        logger.info(f"🔄 Initializing incremental vector index...")
        logger.info(f"   Embedding model: {embedding_model}")
        logger.info(f"   Embedding dimension: {self.embedder.dimension}")
        
        # Create embedding UDF
        embed_udf = create_embedding_udf(embedding_model)
        
        # Add embeddings column to chunks
        self.indexed_table = chunks_table.select(
            **{col: getattr(chunks_table, col) for col in chunks_table.column_names()},
            embedding=embed_udf(getattr(chunks_table, text_column))
        )
        
        logger.info("✅ Vector index initialized (updates incrementally)")
    
    def get_indexed_table(self) -> pw.Table:
        """Get the table with embeddings added."""
        return self.indexed_table


def create_vector_index(
    chunks_table: pw.Table,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> pw.Table:
    """
    Create an incremental vector index from a chunks table.
    
    This is the main entry point for setting up vector search.
    
    Args:
        chunks_table: Table with chunk_text and metadata
        embedding_model: Sentence-transformers model name
        
    Returns:
        Table with embeddings added, ready for similarity search
    """
    index = IncrementalVectorIndex(chunks_table, embedding_model)
    return index.get_indexed_table()


# Export public API
__all__ = [
    "IncrementalVectorIndex",
    "create_vector_index",
]
