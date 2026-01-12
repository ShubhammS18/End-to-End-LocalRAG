from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

from contextlib import asynccontextmanager
import os
from src import config
from src.retriever import Retriever
from src.llm import generate_answer



# Lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not os.path.exists(config.FAISS_INDEX_PATH):
        raise RuntimeError(
            "FAISS index not found. Please ingest documents first."
        )

    app.state.retriever = Retriever()
    print("Retriever loaded successfully")

    yield  # Application runs here

    # Shutdown (optional cleanup)
    print("Application shutting down")



# App initialization
app = FastAPI(
    title="Local RAG API",
    description="A minimal, local Retrieval-Augmented Generation service",
    version="4.1",
    lifespan=lifespan
)


# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Request / Response models
class AskRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K

class Citation(BaseModel):
    rank: int
    text: str

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]

# Core RAG endpoint
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest, req: Request):
    retriever = req.app.state.retriever

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    results = retriever.retrieve(
        query=question,
        top_k=request.top_k
    )
    context_chunks = [r["text"] for r in results]
    answer = generate_answer(context_chunks, question)
    
    citations = [
    {
        "rank": i + 1,"text": chunk}
    for i, chunk in enumerate(context_chunks) ]
    
    return AskResponse(
    answer=answer,
    citations=citations
    )
