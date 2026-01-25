from typing import List, Optional


def calculate_overall_score(scores: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Calculate weighted average of scores.

    Args:
        scores: List of individual scores
        weights: Optional list of weights (must match scores length). If None, uses equal weights.

    Returns:
        Weighted average score

    Raises:
        ValueError: If scores and weights have different lengths or weights don't sum to 1
    """
    if not scores:
        return 0.0

    if weights is None:
        # Equal weights for all scores
        return sum(scores) / len(scores)

    if len(scores) != len(weights):
        raise ValueError(
            f"Scores ({len(scores)}) and weights ({len(weights)}) must have same length")

    if not abs(sum(weights) - 1.0) < 1e-6:  # Allow small floating point error
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    return sum(score * weight for score, weight in zip(scores, weights))
