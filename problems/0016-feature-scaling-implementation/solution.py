import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	


	mx =np.max(data,axis=0,keepdims=True)
	mn=np.min(data,axis=0,keepdims=True)
	normalized_data=(data-mn)/(mx-mn)

	mean=np.mean(data,axis=0,keepdims=True)
	std = np.std(data, axis=0, keepdims=True)
	standardized_data=(data-mean)/std

	return standardized_data, normalized_data