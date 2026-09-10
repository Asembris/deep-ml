import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    
    N = preds.shape[0]

    eps = 1e-12
    safe_preds = np.clip(preds, eps, 1.0)

    correct_probs = safe_preds[np.arange(N), target]


    losses = -np.log(correct_probs)


    grad = np.zeros_like(preds)
    grad[np.arange(N), target] = -1.0 / correct_probs

    if reduction == "mean":
        loss = np.mean(losses)
        grad /= N

    elif reduction == "sum":
        loss = np.sum(losses)

    elif reduction == "none":
        loss = losses

    else:
        raise ValueError(
            f"Invalid reduction '{reduction}'. "
            "Expected 'mean', 'sum', or 'none'."
        )

    return loss, grad