import json

from src.retriever import Retriever
from src.confidence import evaluate_confidence, ConfidenceDecision


def run_evaluation():
    # Load evaluation questions
    with open("eval/eval_questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    retriever = Retriever()

    total = 0
    correct = 0
    false_allow = 0
    false_refuse = 0

    print("\n=== Running RAG Confidence Evaluation ===\n")

    for item in data:
        question = item["question"]
        expected_answerable = item["answerable"]

        results = retriever.retrieve(question)
        confidence = evaluate_confidence(results)

        decision = confidence["decision"]

        predicted_answerable = decision != ConfidenceDecision.REFUSE

        total += 1

        if predicted_answerable == expected_answerable:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
            if predicted_answerable and not expected_answerable:
                false_allow += 1
            if not predicted_answerable and expected_answerable:
                false_refuse += 1

        print(f"{status} | {decision.upper():6} | {question}")

    print("\n=== Evaluation Summary ===")
    print(f"Total questions     : {total}")
    print(f"Correct decisions   : {correct}")
    print(f"False allows (bad)  : {false_allow}")
    print(f"False refuses (OK)  : {false_refuse}")
    print("==========================\n")


if __name__ == "__main__":
    run_evaluation()
