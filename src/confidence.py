from typing import List, Dict
from src import config


class ConfidenceDecision:
    """
    Possible confidence decisions for retrieval quality.
    """
    ALLOW = "allow"
    WARN = "warn"
    REFUSE = "refuse"


def evaluate_confidence(results: List[Dict]) -> Dict:
    """
    Evaluate retrieval results and decide whether it is safe to answer.

    Returns a structured decision dict:
    {
        "decision": "allow" | "warn" | "refuse",
        "reason": str
    }
    """

    # Rule 1: No retrieval
    if not results:
        return {
            "decision": ConfidenceDecision.REFUSE,
            "reason": "No relevant context retrieved."
            }

    
    # Extract similarity scores
    scores = [r["score"] for r in results]

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    
    
    # Rule 2: Very weak best match
    if max_score < config.MIN_CONFIDENCE_SCORE:
        return {
            "decision": ConfidenceDecision.REFUSE,
            "reason": (
                f"Retrieval confidence too low "
                f"(max score = {max_score:.3f})."
            )
                }

    # Rule 3: Weak overall context
    if avg_score < config.MIN_AVG_SCORE:
        return {
            "decision": ConfidenceDecision.WARN,
            "reason": (
                f"Weak overall context quality "
                f"(avg score = {avg_score:.3f})."
            )
                }

    # Rule 4: Safe to answer
    return {
        "decision": ConfidenceDecision.ALLOW,
        "reason": "Sufficient retrieval confidence."
            }
