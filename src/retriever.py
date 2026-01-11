import pickle
from typing import List, Dict
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from . import config




# -------------------------
# Retrieval logic
# -------------------------

class Retriever:
    def __init__(self):
        if not os.path.exists(config.FAISS_INDEX_PATH):
            raise FileNotFoundError(
                "FAISS index not found. Please ingest documents first."
            )

        # Load FAISS index
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)

        # Load chunks (index-aligned)
        with open(config.CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        
        # Load metadata (index-aligned)
        with open(config.METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
            
        # Loading embedding model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[Dict]:
        """
        Retrieve top-k most relevant text chunks for a query.
        """
        # 1. Embedding the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # 2. Normalize query (cosine similarity via inner product)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # 3. FAISS search
        scores, indices = self.index.search(
            query_embedding.astype("float32").reshape(1, -1),
            top_k
        )
        
        # 4. Build structured results
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx]
                })
        return results