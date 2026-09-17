import numpy as np

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
	"""
	Apply a transformer block with two sublayers using either pre-norm or post-norm.
	
	Args:
		x: Input array of shape (seq_len, d_model)
		W1, b1: Weights and bias for first sublayer
		W2, b2: Weights and bias for second sublayer
		gamma1, beta1: LayerNorm params for first normalization
		gamma2, beta2: LayerNorm params for second normalization
		mode: 'pre_norm' or 'post_norm'
		eps: Epsilon for numerical stability
	
	Returns:
		Output array of shape (seq_len, d_model)
	"""
	def layer_norm(x: np.ndarray,gamma:np.ndarray,beta:np.ndarray):
		mean=np.mean(x,axis=-1,keepdims=True)
		var=np.var(x,axis=-1,keepdims=True)
		x_norm=(x-mean)/(np.sqrt(var+eps))
		y=gamma * x_norm + beta
		return y 
	def sublayer(x,w,b):
		return x @ w + b 
	
	if mode=="pre_norm":
		h=x+sublayer(layer_norm(x,gamma1,beta1),W1,b1)
		out=h+sublayer(layer_norm(h,gamma2,beta2),W2,b2)
		return out
	else:
		h=layer_norm(x+sublayer(x,W1,b1),gamma1,beta1)
		out=layer_norm(h+sublayer(h,W2,b2),gamma2,beta2)
		return out




