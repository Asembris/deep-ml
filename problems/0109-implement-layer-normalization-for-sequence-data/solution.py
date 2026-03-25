import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	b,seq_len,d_model=X.shape
	mean=np.sum(X,axis=-1,keepdims=True)/d_model
	var=np.var(X,axis=-1,keepdims=True)
	X_n=(X-mean)/np.sqrt(var+epsilon)
	Y_n=gamma*X_n+beta
	return Y_n
	pass