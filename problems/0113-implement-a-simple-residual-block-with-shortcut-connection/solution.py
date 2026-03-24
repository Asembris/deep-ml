import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	
	x1=np.dot(w1,x)
	x1=np.maximum(0.0,x1)
	x2=np.dot(w2,x1)
	x2=np.maximum(0.0,x2)
	res=x2+x
	res=np.maximum(res,0.0)
	return res
	pass