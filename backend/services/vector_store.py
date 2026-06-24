import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from backend.config import settings

class FAISSVectorStore:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.index_path = os.path.join(settings.INDEX_DIR, "faiss_index.bin")
        self.metadata_path = os.path.join(settings.INDEX_DIR, "metadata.pkl")
        
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Vector dimension: {self.dimension}")
        
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        """Loads vector store and metadata from disk if they exist, otherwise creates a new index."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                print("Loading existing FAISS index...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.chunks = pickle.load(f)
                print(f"FAISS index loaded. Total items: {len(self.chunks)}")
            except Exception as e:
                print(f"Error loading index, creating a new one: {str(e)}")
                self._init_empty_index()
        else:
            self._init_empty_index()

    def _init_empty_index(self):
        print("Initializing empty FAISS index (IndexFlatIP for cosine similarity with L2 normalized vectors)...")
        # flat L2 or Inner Product index
        # Standard cosine similarity is achieved by normalizing vectors and using Inner Product index (IndexFlatIP)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def save(self):
        """Saves current index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.chunks, f)
            print("FAISS index and metadata successfully saved.")

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Encodes chunks to embeddings, normalizes them for cosine similarity, and adds to FAISS index."""
        if not new_chunks:
            return
            
        texts = [chunk["text"] for chunk in new_chunks]
        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Convert to float32 numpy array
        embeddings = np.array(embeddings).astype("float32")
        
        # L2 normalization converts Inner Product search into Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Save metadata
        # Store original chunks but add current index offset
        start_id = len(self.chunks)
        for idx, chunk in enumerate(new_chunks):
            self.chunks.append({
                "global_id": start_id + idx,
                "text": chunk["text"],
                "start_idx": chunk.get("start_idx"),
                "end_idx": chunk.get("end_idx"),
                "file_name": chunk.get("file_name", "unknown")
            })
            
        self.save()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Queries FAISS index and returns matching chunks with similarity scores."""
        if self.index is None or len(self.chunks) == 0:
            return []
            
        # Encode query
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx].copy()
            # Cosine similarity score ranges from -1 to 1 (usually 0 to 1 for normalized vectors)
            chunk["score"] = float(score)
            results.append(chunk)
            
        return results

    def clear(self):
        """Clears all records from the index and deletes storage files."""
        self._init_empty_index()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        print("FAISS vector store cleared successfully.")
