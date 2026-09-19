import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	res=np.mean(y_pred==y_true)
	return res