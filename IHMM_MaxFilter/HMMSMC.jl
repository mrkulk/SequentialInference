


using Distributions
using Debug
using PyCall
#require("pmapLoading.jl")
# @pyimport numpy.random as nr
# println(nr.rand(3,4))
# @pyimport scipy.optimize as so
# so.newton(x -> cos(x) - x, 1)
# @pyimport matplotlib.pylab as plt

# plt.plot(x, y)
# plt.savefig("foo.png", bbox_inches=0)
#plt.show() 
@debug begin

##################PARAMETERS###################
myappend{T}(v::Vector{T}, x::T) = [v..., x]
NUM_PARTICLES_SAMP = 10


transition_matrix = [[0.1, 0.2, 0.3, 0.4] [0.25, 0.25, 0.25, 0.25] [0.4, 0.3, 0.2, 0.1] [0.2, 0.2, 0.2, 0.4]]
emission_matrix = [[0.2, 0.3, 0.5] [0.25, 0.25, 0.5] [0.5, 0.1, 0.4] [0.2, 0.7, 0.1]]
initial_matrix = [0.25, 0.25, 0.25, 0.25]
obs_sequence = Dict()

seq_true = [2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,1.0,1.0,2.0,1.0,1.0]
obs = [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0,1.0,1.0,2.0,0.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,2.0,0.0,1.0,2.0,2.0,1.0,1.0,2.0,1.0,1.0]

# seq_true = [2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,4.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# obs = [2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,4.0,4.0,1.0,1.0,1.0,1.0,3.0,1.0,1.0,3.0,1.0,2.0,1.0,4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

# seq_true = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,2.0,2.0,2.0,1.0,1.0,2.0,2.0,1.0]
# obs = [3.0,1.0,1.0,1.0,3.0,1.0,2.0,1.0,2.0,2.0,3.0,3.0,2.0,3.0,3.0,3.0,3.0,1.0,3.0,3.0,4.0,3.0,2.0,2.0,2.0,1.0,1.0,2.0,2.0,1.0]

SEQUENCE_LENGTH = length(seq_true)
for j = 1:SEQUENCE_LENGTH
	obs_sequence[j] = obs[j]
end



#################HELPER FUNCTIONS#################

##########GENERATING DATA FUNCTIONS


function transition(currentState, transition_mat)
	#println(currentState)
	sample_arr = rand(Multinomial(1, (transition_mat)'[:, currentState]))
	nxtState = findin(sample_arr, 1)[1]
	return nxtState
end


function emission(currentState, emission_mat)
	@bp
	sample_arr = rand(Multinomial(1, (emission_mat)'[:, currentState]))
	currentObs = findin(sample_arr, 1)[1]
	return currentObs
end

function initialize()
	sample_arr = rand(Multinomial(1, initial_matrix))
	nxtState = findin(sample_arr, 1)[1]
	return nxtState
end



######################EXACT MARGINAL LIKELIHOOD (scaled)


function compExactMarginal(transition_matrix, emission_matrix, init_state, obs_seq, scaled)
	println(scaled)
	obs_len = length(obs_seq)
	num_states = size(transition_matrix)[1]
	alpha_mat = zeros(obs_len, num_states)
	scaled_alpha_mat = zeros(obs_len, num_states)
	c_vect = zeros(length(obs_seq))
	
	alpha_mat[1, :] = transition_matrix[init_state, :] .* emission_matrix[:, obs_seq[1]]'
	c_vect[1] = sum(alpha_mat[1, :])
	scaled_alpha_mat[1, :] = alpha_mat[1, :] ./ c_vect[1]

	if scaled == false
		for i = 2:obs_len
			
			for j = 1 : num_states
				alpha_mat[i, j] = 0
				temp_sum = 0
				for k = 1: num_states
					temp_sum += alpha_mat[i - 1, k] * transition_matrix[k, j]
				end 
				alpha_mat[i, j] = temp_sum * emission_matrix[j, obs_seq[i]]
			end
		end

		marginal = sum(alpha_mat[obs_len, :])
	else

		for i = 2:obs_len
			
			for j = 1 : num_states
				scaled_alpha_mat[i, j] = 0
				temp_sum = 0
				for k = 1: num_states
					temp_sum += scaled_alpha_mat[i - 1, k] * transition_matrix[k, j]
				end 
				alpha_mat[i, j] = temp_sum * emission_matrix[j, obs_seq[i]]
			end
			c_vect[i] = sum(alpha_mat[i, :])
			
			println("c ", c_vect[i])
			println("obs: ", obs_seq[i])
			println("i ", i)
			println()
			scaled_alpha_mat[i, :] = alpha_mat[i, :] ./ c_vect[i]
		end

		marginal = sum(log(c_vect))
		marginal = (marginal)
	end

	return marginal
end


############## PARTICLE FUNCTIONS

function initPartilce(particle, obsSeq, emission_mat, init_state, transition_mat)
	# currentSize = length(keys(particle))
	newObs = obsSeq[1]
	particle[1] = {"weight" => 0, "hidden_state" => 0}
	sampledState = transition(init_state, transition_mat)
	particle[1]["hidden_state"] = sampledState
	weight = (emission_mat)[:, sampledState][newObs]
	logWeight = log(weight)
	particle[1]["weight"] = logWeight

end

function extendParticle(particle, obsSeq, emission_mat, transition_mat)
		
	currentSize = length(keys(particle))
	newObs = obsSeq[currentSize + 1]
	particle[currentSize + 1] = {"weight" => 0, "hidden_state" => 0}
	currentState = particle[currentSize]["hidden_state"]
	sampledState = transition(currentState, transition_mat)
	particle[currentSize + 1]["hidden_state"] = sampledState[1]
	
	println("OBS:")
	println(sampledState)

	weight = (emission_mat)[:, sampledState][newObs]
	#println("sampledState ", sampledState, " obs ", newObs)
	logWeight = log(weight)
	
	particle[currentSize + 1]["weight"] = logWeight
end

function normalizeWeight(particlesDict)
	timePoint = length(keys(particlesDict[1]))
	weightVect = zeros(NUM_PARTICLES_SAMP)
	for k = 1:NUM_PARTICLES_SAMP
		
		weightVect[k] = exp(particlesDict[k][timePoint]["weight"])
		#println(particlesDict[k][timePoint]["weight"])
	end
	vectSum = sum(weightVect)
	if vectSum != 0
		normalizeWeightVect = weightVect / vectSum
		avg_weight = vectSum / NUM_PARTICLES_SAMP
	else
		normalizeWeightVect = ones(NUM_PARTICLES_SAMP) ./ NUM_PARTICLES_SAMP
		avg_weight = 1 / NUM_PARTICLES_SAMP
	end
	return normalizeWeightVect, avg_weight
end





function resampleParticles(particlesDict)
	newIndices = zeros(NUM_PARTICLES_SAMP)
	tempWeightVect = normalizeWeight(particlesDict)
	normalizeWeightVect = tempWeightVect[1]

	avg_weight = tempWeightVect[2]
	particles_temporary = deepcopy(particlesDict)
	for i = 1:NUM_PARTICLES_SAMP
		#println(normalizeWeightVect)
		sample_arr = rand(Multinomial(1,normalizeWeightVect))
		newIndices[i] = findin(sample_arr, 1)[1]
		particlesDict[i] = deepcopy(particles_temporary[newIndices[i]])
	end
	return avg_weight
end




function approxMarginalLikelihood(obsSeq, transition_mat, emission_mat, init_state)
	log_weight_vect = zeros(length(obsSeq))
	LENGTH_SEQ = length(obsSeq)
	particles = Dict()
	for i = 1:NUM_PARTICLES_SAMP
		particles[i] = Dict()
	end

	for k = 1:NUM_PARTICLES_SAMP
		
			initPartilce(particles[k], obsSeq, emission_mat, init_state, transition_mat)
	end
	
	temp_avg_weight = resampleParticles(particles)
	
	log_weight_vect[1] = temp_avg_weight

	for l = 2:LENGTH_SEQ
		for k = 1:NUM_PARTICLES_SAMP
				extendParticle(particles[k], obsSeq, emission_mat, transition_mat)
		end	
		log_weight_vect[l] = resampleParticles(particles)

	end
	log_marginal_likelihood = sum(log(log_weight_vect))
	marginal_likelihood = exp(log_marginal_likelihood)
	finalWeightVect = normalizeWeight(particles)
	return (log_marginal_likelihood)
end
	

approxMarginalLikelihood(obs_sequence, transition_matrix, emission_matrix, 1)




end #debug 


