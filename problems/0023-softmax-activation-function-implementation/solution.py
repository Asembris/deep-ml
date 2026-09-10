import math

def softmax(scores: list[float]) -> list[float]:
    mx=max(scores)
    scores=[math.exp(e-mx) for e in scores]
    res=[e/sum(scores) for e in scores]
    return res
    