import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	d=K.shape[1]
	att_scores=np.dot(Q,K.T)/np.sqrt(d)
	att_scores+=mask
	att_scores-=np.max(att_scores,axis=1,keepdims=True)
	att_weights=np.exp(att_scores)/np.sum(np.exp(att_scores),axis=1,keepdims=True)
	res=np.dot(att_weights,V)
	return res
	pass