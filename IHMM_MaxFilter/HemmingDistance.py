



from munkres import Munkres, print_matrix


def teest():
	matrix = [[5, 9, 1],
	          [10, 3, 2],
	          [8, 7, 4]]
	m = Munkres()
	indexes = m.compute(matrix)
	print_matrix(matrix, msg='Lowest cost through this matrix:')
	total = 0
	for row, column in indexes:
	    value = matrix[row][column]
	    total += value
	return total
