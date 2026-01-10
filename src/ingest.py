import os
import pickle
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import config


# -------------------------
# Text loading utilities
# -------------------------

def load_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_text_from_txt(file_path: str) -> str:
    """Load text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------
# Chunking logic
# -------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        # Drop empty or very small chunks
        if len(chunk) >= config.MIN_CHUNK_LENGTH:
            chunks.append(chunk)
        
        chunks.append(chunk)
        start = end - overlap

        if start < 0:
            start = 0

    return chunks


# -------------------------
# Ingestion pipeline
# -------------------------

def ingest_documents(file_path: str) -> None:
    """
    Load document, chunk text, generate embeddings,
    and store them locally for retrieval.
    """
    # 1. Loading text
    if file_path.endswith(".pdf"):
        text = load_text_from_pdf(file_path)
    elif file_path.endswith(".txt"):
        text = load_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or TXT.")

    # 2. Chunking text
    chunks = chunk_text(
        text,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )

    # 3. Loading embedding model
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    # 4. Generating embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    # 5. Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 6. Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.astype("float32"))
    
    # 7. Save index + chunks
    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, config.FAISS_INDEX_PATH)

    with open(config.CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Ingestion complete. Stored {len(chunks)} chunks.")


# -------------------------
# CLI entry point
# -------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_document>")
        sys.exit(1)

    ingest_documents(sys.argv[1])
