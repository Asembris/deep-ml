import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    log_probs=np.array(log_probs)
    correct_answers=np.array(correct_answers)
    predictions=np.argmax(log_probs,axis=-1)

    shifted = log_probs - np.max(log_probs, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    avg_correct_prob=np.mean(probs[np.arange(len(correct_answers)), correct_answers])
    accuracy=np.mean(predictions==correct_answers)

    res={"accuracy":np.round(accuracy,4),"predictions":predictions.tolist(),"avg_correct_prob":np.round(avg_correct_prob,4)}

    return res


