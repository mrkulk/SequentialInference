


using Distributions
using Debug
using PyCall
#require("BasicHerding.jl")
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


LENGTH_SEQ = 50

transition_matrix = [[0.8, 0.1, 0.05, 0.05] [0.05, 0.8, 0.1, 0.05] [0.05, 0.05, 0.8, 0.1] [0.1, 0.05, 0.05, 0.8]]
emission_matrix = [[0.8, 0.1, 0.05, 0.05] [0.05, 0.8, 0.1, 0.05] [0.05, 0.05, 0.8, 0.1] [0.1, 0.05, 0.05, 0.8]]
initial_matrix = [0.25, 0.25, 0.25, 0.25]




##########GENERATING DATA FUNCTIONS

function transition(currentState)
	sample_arr = rand(Multinomial(1, (transition_matrix)[:, currentState]))
	nxtState = findin(sample_arr, 1)[1]
	return nxtState
end


function emission(currentState)
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

function main_generate(seed)

srand(seed)
hidden_state_seq = zeros(LENGTH_SEQ)
observation_seq = zeros(LENGTH_SEQ)

hidden_state_seq[1] = initialize()
observation_seq[1] = emission(hidden_state_seq[1])
for t = 2:LENGTH_SEQ
	hidden_state_seq[t] = transition(hidden_state_seq[t-1])
	observation_seq[t] = emission(hidden_state_seq[t])
end


println(hidden_state_seq)
println(observation_seq)
return {"hid" => hidden_state_seq, "obs" => observation_seq}
end




end #debug 


