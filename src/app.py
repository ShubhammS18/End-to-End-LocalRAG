import os
import streamlit as st

from ingest import ingest_documents
from retriever import Retriever
from llm import generate_answer
import config


# -------------------------
# App configuration
# -------------------------

st.set_page_config(page_title="Minimal Local RAG", layout="wide")

st.title("📄 Minimal Local RAG System")
st.write("Upload a document and ask questions using local LLM + embeddings.")


# -------------------------
# Sidebar: document upload
# -------------------------

st.sidebar.header("📂 Document Ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    temp_path = os.path.join("temp_upload", uploaded_file.name)
    os.makedirs("temp_upload", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.sidebar.button("Ingest Document"):
        with st.spinner("Ingesting document..."):
            ingest_documents(temp_path)
        st.sidebar.success("Document ingested successfully.")


# -------------------------
# Main: Question answering
# -------------------------

st.header("❓ Ask a Question")

question = st.text_input("Enter your question:")

if question:
    if not os.path.exists(config.VECTOR_STORE_PATH):
        st.warning("Please ingest a document first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            retriever = Retriever()
            context_chunks = retriever.retrieve(question)
            answer = generate_answer(context_chunks, question)

        st.subheader("💬 Answer")
        st.write(answer)

        with st.expander("📚 Retrieved Context"):
            for i, chunk in enumerate(context_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)