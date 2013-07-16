
function dataset1()
	"""theta = {
	[1,0,0],
	[0,1,0],
	[0,0,1]}"""
	
	theta={
	[1/3,2/3,0], 
	[4/5,1/10,1/10], 
	[1/8,1/8,6/8]} #3 states"""


	pi = [1/3,1/3,1/3] 
	
	V = 3
	NUM_TOPICS = 3
	return theta, pi, NUM_TOPICS, V
end

