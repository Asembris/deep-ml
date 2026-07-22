import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	v1=np.array(v1)
	v2=np.array(v2)
	prod=np.dot(v1,v2)
	m_v1=np.linalg.norm(v1)
	m_v2=np.linalg.norm(v2)
	res=prod/(m_v1*m_v2)
	return res
	pass