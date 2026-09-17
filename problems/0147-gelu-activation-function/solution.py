import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	

	erf=np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x*x*x))
	scores=0.5 * x * (1 + erf)
	return scores