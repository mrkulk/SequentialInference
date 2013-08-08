

using Distributions
using Debug
using PyCall
using NumericExtensions
require("pmapLoading.jl")
# @pyimport numpy.random as nr
# println(nr.rand(3,4))
# @pyimport scipy.optimize as so
# so.newton(x -> cos(x) - x, 1)
 @pyimport matplotlib.pylab as plt

# plt.plot(x, y)
# plt.savefig("foo.png", bbox_inches=0)
#plt.show() 
@debug begin

##################PARAMETERS###################
if length(ARGS) > 0
	NUM_PARTICLES = int(ARGS[1])
else
	NUM_PARTICLES = 50#1
end

myappend{T}(v::Vector{T}, x::T) = [v..., x]
# NUM_PARTICLES = 50
LENGTH_SEQ = 100
OBS_VAR = 1
TRANSITION_VAR = 10
INIT_VAR = sqrt(5)
NUM_SEQ = 10
SequnceDict = Dict()
hiddenDict = Dict()
NUM_ITERATIONS = 10
PARALELLIZATION = 0
#LOOKAHEAD = false
NUM_BRANCHES, DEPTH = 2, 4



################ HELPER FUNCTIONS #################

################GENERATING DATA FUNCTIONS
#srand(2)
for i = 1:NUM_SEQ
	SequnceDict[i] = Dict()
	hiddenDict[i] = Dict()
end


function transition(currentState, time)
	n = time
	X_n_1 = currentState
	X_n = X_n_1 / 2 + (25 * (X_n_1 / (1 + X_n_1 ^ 2))) + 8 * cos(1.2 * n)
	newState = rand(Normal(X_n, TRANSITION_VAR), 1)
	pdf_func = Normal(X_n, TRANSITION_VAR)
	logWeight = log(pdf(pdf_func, newState))
return [newState[1], logWeight]
end


function emission(currentState)
	#@bp
	X_n = currentState
	Y_n = (X_n ^ 2) / 20 
	newObs = rand(Normal(Y_n, OBS_VAR), 1) 
	pdf_func = Normal(Y_n, OBS_VAR)
	logWeight = log(pdf(pdf_func, newObs))
return [newObs[1], logWeight]
end


function initSequnece()
	X_1 = rand(Normal(0, INIT_VAR), 1)
	pdf_func = Normal(0, INIT_VAR)
	logWeight = log(pdf(pdf_func, X_1))
	return [X_1[1], logWeight]
end

############## PARTICLE FUNCTIONS

function initPartilce(particle, obsSeq)
	# currentSize = length(keys(particle))
	newObs = obsSeq[1]
	particle[1] = {"weight" => 0, "hidden_state" => 0}
	sampledState = initSequnece()
	particle[1]["hidden_state"] = sampledState[1]
	X_n = sampledState[1]
	Y_n = (X_n ^ 2) / 20
	pdf_func = Normal(Y_n, OBS_VAR)
	logWeight = log(pdf(pdf_func, newObs))
	particle[1]["weight"] = logWeight

end


function recursiveLookahead(currentState, currentProb, currentDepth, numBranches, depth, obsSeq, currentTime)
	
	if currentDepth >= depth
		#println("targetDepth ", depth)
		#@bp
		return 0
	else
		#@bp
 		nxtState = zeros(numBranches)
 		probOfBrnach = zeros(numBranches)
 		tempProb = 0

 		currentDepth += 1
 		currentTime += 1
 		# println()
 		# println("now entered new node")
 		#println("depth ", currentDepth, " currentTime ", currentTime)
 		newObs = obsSeq[currentTime]
 		branches = [1:numBranches]
 		while ~isempty(branches)
 			#@bp
 			i = branches[1]
 			shift!(branches)
 			#println("currentTime ", currentTime, " currentState ", currentState, " depth ", currentDepth, " branch ", i)
 			sampledState = transition(currentState, currentTime - 1)

 			nxtState[i] = sampledState[1]
 			X_n = sampledState[1]
			Y_n = (X_n ^ 2) / 20
			pdf_func = Normal(Y_n, OBS_VAR)
			obsLogProb = log(pdf(pdf_func, newObs))
			if obsLogProb == -Inf
				obsLogProb = typemin(Int32)
			end
 			logProb = sampledState[2] + obsLogProb

 			#currentProb = logProb
 			prob = recursiveLookahead(nxtState[i], currentProb, currentDepth, numBranches, depth, obsSeq, currentTime)
 			probOfBrnach[i] = prob + logProb
 			# currentDepth -= 1
 			# currentTime -= 1
 			#println("prob ", prob, " currentDepth ", currentDepth, " currentTime ", currentTime)
 			# tempProb += currentProb
 			# tempProb += prob
 			#println("now in branch ", i)
 			#println("i: ", i, " branches ", branches)
 		end
 		probOfAllBrnaches = logsumexp(probOfBrnach)

 		tempProb = probOfAllBrnaches
 		# println()
 		# println(" branches ", branches)
 		# println("temp Prob ", tempProb)
 		# println()
 		currentDepth -= 1
 		currentTime -= 1
 		return tempProb
 	end

end




function extendParticle(particle, obsSeq, lookahead)
	currentSize = length(keys(particle))
	newObs = obsSeq[currentSize + 1]
	particle[currentSize + 1] = {"weight" => 0, "hidden_state" => 0}
	currentState = particle[currentSize]["hidden_state"]
	sampledState = transition(currentState, currentSize)
	particle[currentSize + 1]["hidden_state"] = sampledState[1]
	X_n = sampledState[1]
	Y_n = (X_n ^ 2) / 20
	pdf_func = Normal(Y_n, OBS_VAR)
	logWeight = log(pdf(pdf_func, newObs))
	if logWeight == -Inf
				logWeight = typemin(Int32)
	end
	if lookahead == true
		depth = min(DEPTH, (LENGTH_SEQ - currentSize))
		logWeight = recursiveLookahead(sampledState[1], logWeight, 0, NUM_BRANCHES, depth, obsSeq, currentSize)
	end
	particle[currentSize + 1]["weight"] = logWeight
end



function normalizeWeight(particlesDict)
	timePoint = length(keys(particlesDict[1]))
	weightVect = zeros(NUM_PARTICLES)
	#println(particlesDict)
	for k = 1:NUM_PARTICLES
		#the weights are n log format
		weightVect[k] = (particlesDict[k][timePoint]["weight"])
	end

	vectSum = logsumexp(weightVect)
	#println(weightVect)
	normalizeWeightVect = weightVect - vectSum
	# println("weigthVect ", weightVect)
	# println("vectSum ", vectSum)
	# println("lognormalizeWeigth ", normalizeWeightVect)
	expWeight = exp(normalizeWeightVect)
	ess = 1 / sum(expWeight .^ 2)
	#println("ess: ", ess)
	return expWeight
end





function resampleParticles(particlesDict)
	newIndices = zeros(NUM_PARTICLES)
	normalizeWeightVect = normalizeWeight(particlesDict)
	
	particles_temporary = copy(particlesDict)

	if PARALELLIZATION == 0
		# println("particle weight")
		 #println(normalizeWeightVect)
		for i = 1:NUM_PARTICLES
			
			sample_arr = rand(Multinomial(1,normalizeWeightVect))
			newIndices[i] = findin(sample_arr, 1)[1]
			particlesDict[i] = copy(particles_temporary[newIndices[i]])
		end
		#println()
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





################GENERATE DATA##################



for k = 1:NUM_SEQ
	currentState = initSequnece()
	obs = emission(currentState[1])
	SequnceDict[k][1] = obs[1]
	hiddenDict[k][1] = currentState[1]
	for j = 2:LENGTH_SEQ
		currentState = transition(currentState[1], j)
		obs = emission(currentState[1])
		hiddenDict[k][j] = currentState[1]
		SequnceDict[k][j] = obs[1]
	end
end

x = zeros(LENGTH_SEQ)
z = zeros(LENGTH_SEQ)
for ii = 1:LENGTH_SEQ
	x[ii] = SequnceDict[1][ii]
	z[ii] = hiddenDict[1][ii]
end
# y = 1:LENGTH_SEQ
#  plt.plot(y, x)
#  plt.plot(y, z, color = "red")
 # plt.savefig("foo.png", bbox_inches=0)





###################### SMC ######################


obsSeq = SequnceDict[1]
hiddenSeq = hiddenDict[1]

hiddenStatesMatrix = reshape(zeros(NUM_ITERATIONS * LENGTH_SEQ), NUM_ITERATIONS, LENGTH_SEQ)


function runExp(lookahead)

	for iter = 1:NUM_ITERATIONS
	#println("iter", iter)
	# tic()
	particles = Dict()
	for i = 1:NUM_PARTICLES
		particles[i] = Dict()
	end

	for k = 1:NUM_PARTICLES

			initPartilce(particles[k], obsSeq)
			#println("particlesINIT ", particles[k])
	end
	resampleParticles(particles)

	for l = 2:LENGTH_SEQ
		for k = 1:NUM_PARTICLES
				extendParticle(particles[k], obsSeq, lookahead)
		end	
		currentLength = length(keys(particles[1]))
		resampleParticles(particles)
	end
	# toc()

	finalWeightVect = normalizeWeight(particles)
	#println(finalWeightVect)
	sample_arr = rand(Multinomial(1, finalWeightVect))
	finalIndex = findin(sample_arr, 1)[1]
	finalParticle = particles[finalIndex]
	# println("particles", particles[finalIndex])
	tempHiddenStateSeq = [finalParticle[k]["hidden_state"] for k in sort(collect(keys(finalParticle)))]
	hiddenStatesMatrix[iter, :] = tempHiddenStateSeq
end

hiddenSeq = [hiddenSeq[l] for l = 1:LENGTH_SEQ]
tempDiff = reshape(mean(hiddenStatesMatrix, 1), 1, LENGTH_SEQ) - reshape(hiddenSeq, 1, LENGTH_SEQ)
tempSq = tempDiff .^ 2
MSE = (1 / LENGTH_SEQ) * sum(tempSq) 
#println(sqrt(MSE))
return sqrt(MSE)
# for l = 1:NUM_ITERATIONS
# 	for k = 1:LENGTH_SEQ
# 		z[k] = hiddenStatesMatrix[l, :][k]
		
# 	end
# 	# println("z ", z)
# 	#plt.plot(y, z, color = "green")

# end
end

MSEwithoutlookahead = runExp(false)
MSEwithlookahead = runExp(true)

#plt.savefig("foo3.png", bbox_inches=0)


print([MSEwithoutlookahead, MSEwithlookahead])
end #debug










############################ 





