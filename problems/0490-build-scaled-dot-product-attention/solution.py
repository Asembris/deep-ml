import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	dk=K.shape[-1]
	scaled_product=(Q @ K.T)/np.sqrt(dk)
	if mask is not None:
		scaled_product=np.where(mask==0,float("-inf"),scaled_product)
	mx=np.max(scaled_product,axis=-1,keepdims=True)
	attention_weights=np.exp(scaled_product-mx)/np.sum(np.exp(scaled_product-mx),axis=-1,keepdims=True)
	output= attention_weights @ V
	return (output,attention_weights)



