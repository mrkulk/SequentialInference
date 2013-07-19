
function dataset1()
	mu={[0,0], [2,2], [4,4]}
	std={[0.25,0.25], [0.25,0.25], [0.25,0.25]}
	mixture_weight = [1/2,1/6,1/3]
	return mu, std, mixture_weight
end


function dataset2()
	mu={[0,0], [1,1], [2,2]}
	std={[0.3,0.3], [0.3,0.3], [0.3,0.3]}
	mixture_weight = [1/3,1/3,1/3]
	return mu, std,mixture_weight
end
