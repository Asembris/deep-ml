import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
    """
    Compute temperature-scaled softmax probabilities from logits.
    """
    if temperature == 0:
        probs = np.zeros_like(logits, dtype=float)
        probs[np.argmax(logits)] = 1.0
        return probs.tolist()

    scaled = logits / temperature

    # Stable softmax
    scaled = scaled - np.max(scaled)

    exp_logits = np.exp(scaled)
    probs = exp_logits / np.sum(exp_logits)

    return probs.tolist()