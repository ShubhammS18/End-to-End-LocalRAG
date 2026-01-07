import pickle
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config


# -------------------------
# Similarity computation
# -------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computing cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------
# Retrieval logic
# -------------------------

class Retriever:
    def __init__(self):
        # Loading vector store
        with open(config.VECTOR_STORE_PATH, "rb") as f:
            store = pickle.load(f)

        self.embeddings = store["embeddings"]
        self.chunks = store["chunks"]

        # Loading embedding model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[str]:
        """
        Retrieve top-k most relevant text chunks for a query.
        """
        # 1. Embedding the query
        query_embedding = self.model.encode(query)

        # 2. Computing similarity scores
        scores = [
            cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]

        # 3. top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # 4. Returning corresponding chunks
        return [self.chunks[i] for i in top_indices]
