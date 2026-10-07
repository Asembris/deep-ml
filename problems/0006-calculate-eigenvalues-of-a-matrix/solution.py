def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import math
	a,b=matrix[0]
	c,d=matrix[1]
	tr=-(a+d)
	det=a*d-b*c
	detr=tr*tr-4*det
	x1=(-tr+math.sqrt(detr))/2
	x2=(-tr-math.sqrt(detr))/2
	eigenvalues=[x1,x2]
	return eigenvalues