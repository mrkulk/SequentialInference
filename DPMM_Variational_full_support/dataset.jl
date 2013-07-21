
function dataset1()
	"""theta = {
	[1,0,0],
	[0,1,0],
	[0,0,1]}
	"""
	"""theta={
	[1/4,2/4,1/4], 
	[1/3,1/3,1/3], 
	[2/5,2/5,1/5]} #3 states 

	pi = [4/10,3/10,3/10] 
	V = 3
	NUM_TOPICS = 3"""

	V = 20
	theta = Dict()
	for i=1:V
		theta[i] = rand(Dirichlet([2,2,2]))
	end

	pi = rand(Dirichlet(zeros(V)+2))

	return theta, pi, V
end

