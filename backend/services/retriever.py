import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from backend.services.vector_store import FAISSVectorStore

class HybridRetriever:
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.bm25_chunks: List[Dict[str, Any]] = []
        self._init_bm25()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, strip punctuation, split by space."""
        text = text.lower()
        # Keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _init_bm25(self):
        """Initializes BM25 index using all chunks currently in the vector store."""
        chunks = self.vector_store.chunks
        if not chunks:
            self.bm25 = None
            self.bm25_chunks = []
            return

        print(f"Initializing BM25 index with {len(chunks)} chunks...")
        corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.bm25_chunks = chunks
        print("BM25 index initialized.")

    def rebuild_sparse_index(self):
        """Rebuilds the BM25 index, called when new documents are added."""
        self._init_bm25()

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> List[Dict[str, Any]]:
        """Retrieves top chunks using dense, sparse, or hybrid search."""
        if not self.vector_store.chunks:
            return []

        # Ensure BM25 index is synced if chunks count changed
        if len(self.vector_store.chunks) != len(self.bm25_chunks):
            self.rebuild_sparse_index()

        if mode == "dense":
            return self.vector_store.search(query, k=k)
        elif mode == "sparse":
            return self._sparse_search(query, k=k)
        else:  # hybrid
            return self._hybrid_search(query, k=k)

    def _sparse_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Runs BM25 keyword search and returns top chunks."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        # BM25 scores are raw floats (can be negative or large positive)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score in descending order
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            # Exclude negative score terms (usually irrelevant)
            if score <= 0 and len(results) > 0:
                break
            
            chunk = self.vector_store.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
            
        return results

    def _hybrid_search(self, query: str, k: int = 5, rrf_constant: int = 60) -> List[Dict[str, Any]]:
        """Combines BM25 and FAISS results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: RRF_Score(doc) = sum( 1 / (rrf_constant + rank_in_system) )
        """
        # Fetch larger sets of candidates to ensure overlap
        candidate_limit = max(k * 3, 20)
        
        dense_results = self.vector_store.search(query, k=candidate_limit)
        sparse_results = self._sparse_search(query, k=candidate_limit)
        
        # Map global_id to item for reconstruction
        id_to_chunk = {}
        
        # Rank arrays
        dense_ranks = {}
        for rank, chunk in enumerate(dense_results):
            g_id = chunk["global_id"]
            dense_ranks[g_id] = rank + 1  # 1-indexed
            id_to_chunk[g_id] = chunk
            
        sparse_ranks = {}
        for rank, chunk in enumerate(sparse_results):
            g_id = chunk["global_id"]
            sparse_ranks[g_id] = rank + 1  # 1-indexed
            id_to_chunk[g_id] = chunk

        # Compute RRF scores
        rrf_scores = {}
        all_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        
        for g_id in all_ids:
            score = 0.0
            if g_id in dense_ranks:
                score += 1.0 / (rrf_constant + dense_ranks[g_id])
            if g_id in sparse_ranks:
                score += 1.0 / (rrf_constant + sparse_ranks[g_id])
            rrf_scores[g_id] = score

        # Sort global ids by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda g_id: rrf_scores[g_id], reverse=True)[:k]
        
        results = []
        for g_id in sorted_ids:
            chunk = id_to_chunk[g_id].copy()
            chunk["score"] = rrf_scores[g_id]
            # Detail the contribution of each search method for learning insights
            chunk["search_details"] = {
                "dense_rank": dense_ranks.get(g_id, None),
                "sparse_rank": sparse_ranks.get(g_id, None),
                "dense_score": next((c["score"] for c in dense_results if c["global_id"] == g_id), 0.0) if g_id in dense_ranks else 0.0,
                "sparse_score": next((c["score"] for c in sparse_results if c["global_id"] == g_id), 0.0) if g_id in sparse_ranks else 0.0
            }
            results.append(chunk)
            
        return results
