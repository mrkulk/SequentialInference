
function dataset1()
	"""theta = {
	[1,0,0],
	[0,1,0],
	[0,0,1]}
	"""
	theta={
	[1/4,2/4,1/4], 
	[1/3,1/3,1/3], 
	[1/8,1/8,6/8]} #3 states """

	pi = [1/2,1/2,0] 
	V = 3
	NUM_TOPICS = 3
	return theta, pi, NUM_TOPICS, V
end

