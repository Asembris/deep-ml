def softplus(x: float) -> float:
	"""
	Compute the softplus activation function.

	Args:
		x: Input value

	Returns:
		The softplus value: log(1 + e^x)
	"""
	# Your code here
	import numpy as np
	val=np.log(1+np.exp(x))
	return round(val,4)