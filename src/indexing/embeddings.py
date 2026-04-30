"""
Living Documentation Copilot - Local Embeddings

Provides local, free text embeddings using sentence-transformers.
No API keys required - runs entirely on CPU/GPU locally.

This is a wrapper that integrates with Pathway's streaming architecture.
"""

import pathway as pw
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_embedding_model = None
_model_name = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or create the sentence-transformer embedding model.
    
    Uses lazy loading and caching for efficiency.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Popular options:
        - all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
        - all-mpnet-base-v2: Higher quality (768 dimensions)
        - paraphrase-MiniLM-L6-v2: Good for semantic similarity
    """
    global _embedding_model, _model_name
    
    if _embedding_model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"🔄 Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            _model_name = model_name
            logger.info(f"✅ Embedding model loaded (dim={_embedding_model.get_sentence_embedding_dimension()})")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    
    return _embedding_model


def compute_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Compute embedding vector for a single text.
    
    Args:
        text: Text to embed
        model_name: Model to use
        
    Returns:
        List of floats representing the embedding vector
    """
    model = get_embedding_model(model_name)
    
    # Clean the text
    text = text.strip()
    if not text:
        # Return zero vector for empty text
        dim = model.get_sentence_embedding_dimension()
        return [0.0] * dim
    
    # Compute embedding
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def compute_embeddings_batch(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Compute embeddings for multiple texts efficiently.
    
    Args:
        texts: List of texts to embed
        model_name: Model to use
        
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model(model_name)
    
    # Clean texts
    cleaned = [t.strip() if t else "" for t in texts]
    
    # Batch encode
    embeddings = model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Get the dimension of embeddings for a given model."""
    model = get_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


class LocalEmbedder:
    """
    Embedder class compatible with Pathway's interface.
    
    This class wraps sentence-transformers to provide free, local embeddings
    that work with Pathway's incremental vector indexing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._dimension = None
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._dimension = get_embedding_dimension(self.model_name)
        return self._dimension
    
    def __call__(self, text: str) -> List[float]:
        """Embed a single text."""
        return compute_embedding(text, self.model_name)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return compute_embeddings_batch(texts, self.model_name)


# Create a Pathway UDF for embedding
def create_embedding_udf(model_name: str = "all-MiniLM-L6-v2"):
    """
    Create a Pathway UDF for computing embeddings.
    
    Returns:
        pw.udf that takes text and returns embedding vector
    """
    @pw.udf
    def embed_text(text: str) -> List[float]:
        return compute_embedding(text, model_name)
    
    return embed_text


# Export public API
__all__ = [
    "LocalEmbedder",
    "get_embedding_model",
    "compute_embedding",
    "compute_embeddings_batch",
    "get_embedding_dimension",
    "create_embedding_udf",
]
