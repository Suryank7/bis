import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class HybridStore:
    """
    Combines dense vector search (FAISS) with sparse keyword search (BM25)
    using Reciprocal Rank Fusion (RRF) to maximize Hit Rate.
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.encoder = SentenceTransformer(embedding_model_name)
        
        # FAISS Index for Dense Vectors
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        
        # BM25 for Sparse Keyword Search
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Store for actual chunk data
        self.chunks = []
        
    def add_chunks(self, chunks: List[Dict]):
        """
        Indexes the chunks in both FAISS and BM25.
        chunks: List of dicts with 'text' and 'metadata'
        """
        if not chunks:
            return
            
        self.chunks.extend(chunks)
        
        # 1. Index Dense Vectors
        texts = [c['text'] for c in chunks]
        logger.info(f"Computing embeddings for {len(texts)} chunks...")
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        self.faiss_index.add(embeddings)
        
        # 2. Index Sparse Keywords
        logger.info("Building BM25 index...")
        self.tokenized_corpus = [t.lower().split() for t in [c['text'] for c in self.chunks]]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info("Hybrid Store indexing complete.")
        
    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        """
        Retrieves top_k chunks using RRF over Dense and Sparse results.
        """
        if not self.chunks:
            return []
            
        # Dense Retrieval
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        distances, dense_indices = self.faiss_index.search(query_vector, top_k * 2)
        dense_results = dense_indices[0]
        
        # Sparse Retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_param = 60 # standard RRF constant
        
        for rank, idx in enumerate(dense_results):
            if idx != -1:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k_param + rank + 1)
                
        for rank, idx in enumerate(sparse_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k_param + rank + 1)
            
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = [self.chunks[idx] for idx in sorted_indices[:top_k]]
        return results
