from typing import List

import ollama

import config


# -------------------------
# Prompt construction
# -------------------------

def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Building a strict,grounded prompt that minimizes hallucinations 
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
SYSTEM INSTRUCTIONS:
You are an AI assistant answering questions strictly using the provided context.
You must NOT use prior knowledge.
If the answer is not clearly supported by the context, respond with:
"Not found in the provided document."

Context:
{context}

Question:
{question}

Answer(be concise and factual):
""".strip()

    return prompt


# -------------------------
# LLM call
# -------------------------

def generate_answer(context_chunks: List[str], question: str) -> str:
    """
    Generating an answer from the LLM using retrieved context.
    """
    prompt = build_prompt(context_chunks, question)

    response = ollama.chat(
        model=config.OLLAMA_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]
