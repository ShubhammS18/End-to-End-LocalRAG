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
