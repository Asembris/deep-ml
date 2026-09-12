import numpy as np

def square_relu(x: np.ndarray) -> dict:
    output = np.maximum(x, 0) ** 2
    derivative = np.where(x > 0, 2 * x, 0)

    output = np.round(output, 4)
    derivative = np.round(derivative, 4)

    return {
        "output": output,
        "derivative": derivative
    }