from typing import List

import ollama

import config


# -------------------------
# Prompt construction
# -------------------------

def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Building a prompt that instructs the LLM to answer
    strictly using the provided context.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
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
