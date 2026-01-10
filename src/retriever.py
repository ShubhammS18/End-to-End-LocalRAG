import pickle
from typing import List
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config




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
            
        # Loading embedding model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[str]:
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

        # 4. Map indices to chunks
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]