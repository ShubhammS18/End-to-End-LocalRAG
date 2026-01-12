import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# Embedding configuration
# =========================

# SentenceTransformer model used for embedding documents and queries
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Dimensionality of embeddings produced by the model
# all-MiniLM-L6-v2 -> 384
EMBEDDING_DIMENSION = 384


# =========================
# Chunking configuration
# =========================

# Maximum number of characters per text chunk
CHUNK_SIZE = 300

# Number of overlapping characters between consecutive chunks
CHUNK_OVERLAP = 50

# Minimum number of characters for a chunk to be considered valid
MIN_CHUNK_LENGTH = 50


# =========================
# Retrieval configuration
# =========================

# Number of top similar chunks to retrieve for a query
TOP_K = 3


# =========================
# LLM configuration
# =========================

# Ollama model name for local inference
OLLAMA_MODEL_NAME = "llama3.2:1b"


# =========================
# Storage paths
# =========================

# Path where vector store (embeddings + text) will be saved
VECTOR_STORE_PATH = "vector_store.pkl"

# -------------------------
# FAISS vector store paths (v3.2)
# -------------------------

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vector_store", "index.faiss")
CHUNKS_PATH = os.path.join(BASE_DIR, "vector_store", "chunks.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "vector_store", "metadata.pkl")


# Confidence thresholds

MIN_CONFIDENCE_SCORE = 0.35
MIN_AVG_SCORE = 0.30
MIN_RESULTS = 1