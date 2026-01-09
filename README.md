# End-to-End-LocalRAG

# Minimal Local RAG — my-local-rag-v1
 
A minimal, local Retrieval-Augmented Generation (RAG) system I built to learn how retrieval + local LLMs work together.
This repo implements a small, end-to-end pipeline: document ingestion → chunking → embeddings → vector retrieval → grounded generation with a local Ollama model.

### Quick start -

#### create & activate venv (Windows example)
` python -m venv .venv`

`.venv\Scripts\activate `

#### install deps
`pip install -r requirements.txt`

#### ingest a file (from project root)
`python ingest.py path/to/document.pdf`

#### or use the Streamlit app:
`streamlit run app.py`

Open the streamlit UI, upload a PDF/TXT, click Ingest Document, then ask questions in the main UI.

### How it works — 

1. Upload a document (PDF or TXT).

2. The document text is extracted and split into overlapping chunks so each chunk contains meaningful context.

3. Each chunk is converted to a fixed-length vector (embedding) with a SentenceTransformer model. These embeddings are saved to disk.

4. At query time, the user’s question is embedded with the same model. Similarity (cosine) is computed between the question vector and stored chunk vectors. Top-k closest chunks are returned.

5. Those chunks are injected (only those chunks) into a clear prompt that instructs the LLM to answer using only the provided context. If the answer is not in the context, the model replies “Not found in the provided document.”

6. The app shows both the concise answer and the retrieved chunks.

### How I ran tests -

* Created a small sample.txt and verified python ingest.py sample.txt created vector_store.pkl.

* Used a Python shell to import Retriever and confirm retrieve(query) returns semantically relevant chunks.

* Launched streamlit run app.py, ingested a document, asked supported and unsupported questions and validated the grounded response and the “Not found…” refusal.

### v1 → v2 summary -

* v1 — Minimal, working RAG: ingestion, chunking, embeddings, NumPy cosine retrieval, basic prompt. Good for correctness.

* v2 — Focused on retrieval hygiene and grounded prompting:

   * Cleaned chunk generation (strip and drop very short/noisy chunks).

   * Added a configurable MIN_CHUNK_LENGTH.

   * Added safety checks and clamped top_k.

   * Strengthened prompt to explicitly forbid prior knowledge and to refuse if the answer isn’t in context.

Result: fewer hallucinations, more reliable retrieval, clearer demo behavior.

### Known limitations -

* Very small documents can produce duplicate chunks because of overlap and short length (easy to address with deduplication).

* Streamlit session state can retain stale context if the app flow isn’t reset after certain out-of-context queries.

* Vector store is a pickle file (fine for v1, will replace with FAISS/OpenSearch in v3).

* No reranking or citation formatting (v4).


