def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here
	import  math
	def loss_sample(y,p):
		return -(y* math.log(p) + (1-y)*math.log(1-p))

	samples=len(y_true)
	y_clip=[min(max(e,epsilon),1-epsilon) for e in y_pred]

	total_loss=sum([loss_sample(y_true[i],y_clip[i]) for i in range(samples)])/samples

	return total_loss








