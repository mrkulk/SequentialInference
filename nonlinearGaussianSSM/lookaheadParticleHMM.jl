


using Distributions
using Debug
using PyCall
require("pmapLoading.jl")
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
NUM_PARTICLES = 5
LENGTH_SEQ = 10
OBS_VAR = 10
TRANSITION_VAR = 10
INIT_VAR = sqrt(5)
NUM_SEQ = 10
SequnceDict = Dict()
hiddenDict = Dict()
NUM_ITERATIONS = 1
PARALELLIZATION = 0
transition_matrix = [[0.1, 0.2, 0.3, 0.4] [0.25, 0.25, 0.25, 0.25] [0.4, 0.3, 0.2, 0.1] [0.2, 0.2, 0.2, 0.4]]
emission_matrix = [[0.2, 0.3, 0.5] [0.25, 0.25, 0.5] [0.5, 0.1, 0.4] [0.2, 0.7, 0.1]]
initial_matrix = [0.25, 0.25, 0.25, 0.25]

# transition_matrix = [[0.1, 0.2, 0.3, 0.4] [0.1, 0.2, 0.3, 0.4] [0.1, 0.2, 0.3, 0.4] [0.1, 0.2, 0.3, 0.4]]
# emission_matrix = [[0.1, 0.0, 0.9] [0.1, 0.0, 0.9] [0.1, 0.0, 0.9] [0.1, 0.0, 0.9]]
# initial_matrix = [0.25, 0.25, 0.25, 0.25]



#################HELPER FUNCTIONS#################

##########GENERATING DATA FUNCTIONS

function transition(currentState)
	sample_arr = rand(Multinomial(1, (transition_matrix)[:, currentState]))
	nxtState = findin(sample_arr, 1)[1]
	return nxtState
end


function emission(currentState)
	#@bp
	sample_arr = rand(Multinomial(1, (emission_matrix)[:, currentState]))
	currentObs = findin(sample_arr, 1)[1]
	return currentObs
end

function initialize()
	sample_arr = rand(Multinomial(1, initial_matrix))
	nxtState = findin(sample_arr, 1)[1]
	return nxtState
end
#############GENERATE DATA#########################
rand(1)
sequnceDict = Dict()
hiddenDict = Dict()
currentState = initialize()
hiddenDict[1] = currentState
currentObs = emission(currentState)
SequnceDict[1] = currentObs

for i = 2:LENGTH_SEQ
	currentState = transition(currentState)
	currentObs = emission(currentState)
	hiddenDict[i] = currentState
	SequnceDict[i] = currentObs 
end

println(hiddenDict)
println(SequnceDict)

############## PARTICLE FUNCTIONS

function initPartilce(particle, obsSeq)
	# currentSize = length(keys(particle))
	newObs = obsSeq[1]
	particle[1] = {"weight" => 0, "hidden_state" => 0}
	sampledState = initialize()
	particle[1]["hidden_state"] = sampledState[1]
	weight = (emission_matrix)[:, sampledState][newObs]
	logWeight = log(weight)
	particle[1]["weight"] = logWeight

end

function extendParticle(particle, obsSeq)
	currentSize = length(keys(particle))
	newObs = obsSeq[currentSize + 1]
	particle[currentSize + 1] = {"weight" => 0, "hidden_state" => 0}
	currentState = particle[currentSize]["hidden_state"]
	sampledState = transition(currentState)
	particle[currentSize + 1]["hidden_state"] = sampledState[1]
	weight = (emission_matrix)[:, sampledState][newObs]
	#println("sampledState ", sampledState, " obs ", newObs)
	logWeight = log(weight)
	particle[currentSize + 1]["weight"] = logWeight
end

function normalizeWeight(particlesDict)
	timePoint = length(keys(particlesDict[1]))
	weightVect = zeros(NUM_PARTICLES)
	#println(particlesDict)
	for k = 1:NUM_PARTICLES
		weightVect[k] = exp(particlesDict[k][timePoint]["weight"])
	end
	vectSum = sum(weightVect)
	normalizeWeightVect = weightVect / vectSum
	return normalizeWeightVect
end





function resampleParticles(particlesDict)
	newIndices = zeros(NUM_PARTICLES)
	normalizeWeightVect = normalizeWeight(particlesDict)
	
	particles_temporary = copy(particlesDict)

	if PARALELLIZATION == 0
		for i = 1:NUM_PARTICLES
			sample_arr = rand(Multinomial(1,normalizeWeightVect))
			newIndices[i] = findin(sample_arr, 1)[1]
			particlesDict[i] = copy(particles_temporary[newIndices[i]])
		end
	else
		PROCS = 10
		NUM_PARTICLES_BATCHES = [{"s" => int(NUM_PARTICLES/PROCS), "weightVect" => normalizeWeightVect}  for i = 1:PROCS]
		newIndices = pmap(sampleMultinomialVect, NUM_PARTICLES_BATCHES)
		#println(newIndices)
		updateArr = []
		for t in newIndices
			updateArr = vcat(updateArr, t)
	    end

		for j = 1:NUM_PARTICLES
			particlesDict[j] = copy(particles_temporary[updateArr[j]])
		end
	end
end

obsSeq = SequnceDict
hiddenSeq = hiddenDict

hiddenStatesMatrix = reshape(zeros(NUM_ITERATIONS * LENGTH_SEQ), NUM_ITERATIONS, LENGTH_SEQ)

for iter = 1:NUM_ITERATIONS
	println("iter", iter)
	tic()
	particles = Dict()
	for i = 1:NUM_PARTICLES
		particles[i] = Dict()
	end

	for k = 1:NUM_PARTICLES

			initPartilce(particles[k], obsSeq)
	end
	resampleParticles(particles)

	for l = 2:LENGTH_SEQ
		for k = 1:NUM_PARTICLES
				extendParticle(particles[k], obsSeq)
		end	
		#println(particles[1])
		currentLength = length(keys(particles[2]))
		resampleParticles(particles)
	end
	toc()

	finalWeightVect = normalizeWeight(particles)
	println(finalWeightVect)
	sample_arr = rand(Multinomial(1, finalWeightVect))
	finalIndex = findin(sample_arr, 1)[1]
	finalParticle = particles[finalIndex]
	tempHiddenStateSeq = [finalParticle[k]["hidden_state"] for k in keys(finalParticle)]
	hiddenStatesMatrix[iter, :] = tempHiddenStateSeq
end

hiddenSeq = [hiddenSeq[l] for l = 1:LENGTH_SEQ]
tempDiff = reshape(mean(hiddenStatesMatrix, 1), 1, LENGTH_SEQ) - reshape(hiddenSeq, 1, LENGTH_SEQ)
tempSq = tempDiff .^ 2
MSE = (1 / LENGTH_SEQ) * sum(tempSq) 
println(sqrt(MSE))




end #debug 


