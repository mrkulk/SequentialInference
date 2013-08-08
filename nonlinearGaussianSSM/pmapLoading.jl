using Distributions
function sampleMultinomialVect(NUM_PARTICLES_BATCHES)
	NUM_PARTICLES = NUM_PARTICLES_BATCHES["s"]
	normalizeWeightVect = NUM_PARTICLES_BATCHES["weightVect"]
	indices = zeros(NUM_PARTICLES)
	for i = 1:NUM_PARTICLES
		sample_arr = rand(Multinomial(1,normalizeWeightVect))
		indices[i] = findin(sample_arr, 1)[1]
	end
	return indices
end
