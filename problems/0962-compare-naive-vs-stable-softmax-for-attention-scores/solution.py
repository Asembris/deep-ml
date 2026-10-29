import numpy as np

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""

    x = np.array(scores, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        exp_x = np.exp(x)
        naive = exp_x / np.sum(exp_x)

        exp_stable = np.exp(x - np.max(x))
        stable = exp_stable / np.sum(exp_stable)
    if not np.all(np.isfinite(naive)):
        return {
            "naive": [float('nan')] * len(x),
            "stable": [round(float(v), 6) for v in stable],
            "max_abs_diff": float('nan')
        }
    return {
        "naive": [round(float(v), 6) for v in naive],
        "stable": [round(float(v), 6) for v in stable],
        "max_abs_diff": round(float(np.max(np.abs(naive - stable))), 6)
    }

    