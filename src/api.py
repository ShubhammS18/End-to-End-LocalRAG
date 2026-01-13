from fastapi import FastAPI, HTTPException, Request
from httpx import request
from pydantic import BaseModel
from typing import List

from contextlib import asynccontextmanager
import os
from src import config
from src.retriever import Retriever
from src.llm import generate_answer

from src.confidence import evaluate_confidence, ConfidenceDecision

import logging
from datetime import datetime



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

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag-api")


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
    score: float
    source: str | None = None


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
    
    scores = [r["score"] for r in results]

    logger.info(
    "retrieval_metrics | top_k=%d | top_score=%.3f | avg_score=%.3f",
    request.top_k,
    max(scores) if scores else 0.0,
    sum(scores) / len(scores) if scores else 0.0
    )

    
    confidence = evaluate_confidence(results)
    
    logger.info(
    "confidence_decision | decision=%s | reason=%s",
    confidence["decision"],
    confidence["reason"]
    )

    
    # REFUSE: do not call LLM
    if confidence["decision"] == ConfidenceDecision.REFUSE:
        logger.warning(
        "query_refused | question='%s' | reason=%s",
        question[:100],
        confidence["reason"]
        )

        raise HTTPException(
        status_code=422,
        detail={
            "message": "Insufficient context to answer safely.",
            "reason": confidence["reason"]
        }
        )

    context_chunks = [r["text"] for r in results]

    # WARN or ALLOW → LLM is permitted
    answer = generate_answer(context_chunks, question)
    
    logger.info(
    "answer_generated | decision=%s | citations=%d",
    confidence["decision"],
    len(results)
    )


    citations = [
    {
        "rank": r["rank"],
        "text": r["text"],
        "score": r["score"],
        "source": r["metadata"].get("source")
    }
    for r in results
    ]

    
    return AskResponse(
    answer=answer,
    citations=citations
    )

