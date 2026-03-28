from sklearn.metrics import accuracy_score, f1_score


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def evaluate_oos(
    scores: list[float],
    threshold: float,
    total_oos: int,
) -> dict:

    flagged = sum(1 for s in scores if s < threshold)
    return {
        "threshold": threshold,
        "flagged": flagged,
        "total": total_oos,
        "detection_rate": round(flagged / total_oos, 4) if total_oos > 0 else 0.0,
    }