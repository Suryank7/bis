import logging
from typing import List, Dict
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks the top candidates from the Hybrid Store using a Cross-Encoder
    to maximize the MRR @5 metric.
    """
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        logger.info(f"Loading FlashRank Reranker: {model_name}")
        self.ranker = Ranker(model_name=model_name)
        
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Takes the query and retrieved candidates, and reranks them.
        candidates should have 'text' and 'chunk_id' keys.
        """
        if not candidates:
            return []
            
        # Format for FlashRank
        passages = []
        for i, doc in enumerate(candidates):
            passages.append({
                "id": doc.get("chunk_id", str(i)),
                "text": doc.get("text", ""),
                "meta": doc.get("metadata", {})
            })
            
        rank_request = RerankRequest(query=query, passages=passages)
        
        logger.info(f"Reranking {len(candidates)} candidates for query...")
        reranked_results = self.ranker.rerank(rank_request)
        
        # FlashRank returns a list of dictionaries with 'id', 'text', 'score'
        # Re-map them to our original candidate format
        final_results = []
        for res in reranked_results[:top_k]:
            final_results.append({
                "chunk_id": res["id"],
                "text": res["text"],
                "metadata": res.get("meta", {}),
                "rerank_score": res.get("score", 0.0)
            })
            
        return final_results
