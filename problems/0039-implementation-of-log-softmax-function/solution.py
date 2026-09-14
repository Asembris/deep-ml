import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	mx=max(scores)
	scores=np.array(scores)
	scores-=mx 
	soft_max=np.exp(scores)/np.sum(np.exp(scores))
	log_soft=np.log(soft_max)
	return  log_soft