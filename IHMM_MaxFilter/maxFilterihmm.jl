using Distributions
using Debug
using PyCall



srand(1)

type sequence
	suff_stats # a dict from "low" and "top" to the suff stats for these levels
	seq_history
	current_state
end


###################PARAMETERS#####################


TOP_ALPHA = 1
LOW_ALPHA = 1
SEQUENCE_LENGTH = 10







###################HELPER FUNCTIONS###############






#################GENERATIVE MODEL#################


function createCRFProbVect(current_sequence)
	current_state = current_sequence.current_state
	dict_of_current_state = current_sequence.suff_stats["low"][current_state]
	dict_of_state_creation = current_sequence.suff_stats["top"]
	#current_num_states = length(keys(dict_of_current_state)) #only states generated for the current state
	total_num_states = length(keys(dict_of_state_creation)) #all the states generated
	total_num_transitions = sum(values(dict_of_current_state))
	total_num_top_transition = sum(values(dict_of_state_creation))

	prob_vect = zeros(total_num_states * 2 + 1) #one more for the prob of new state creation
	low_temp_denom = LOW_ALPHA + total_num_transitions
	for i = 1:total_num_states
		try
			prob_vect[i] = dict_of_current_state[i] / low_temp_denom
		catch
			prob_vect[i] = 0
		end
	end

	top_temp_denom = TOP_ALPHA + total_num_top_transition
	for i = (total_num_states + 1): (total_num_states * 2)
		
		prob_vect[i] = (LOW_ALPHA / low_temp_denom) * dict_of_state_creation[i - total_num_states] / top_temp_denom
	end

	prob_vect[length(prob_vect)] = (LOW_ALPHA / low_temp_denom) * (TOP_ALPHA / top_temp_denom)
	return prob_vect
end




function updateIHMMSuffStat(current_sequence, sampled_index, time) #use sampled index to find if it's new or old state 
	# suff_stats["low"] = Dict()
	# suff_stats["top"] = Dict()
	# suff_stats["low"][1] = Dict()
	# suff_stats["top"][1] = 1
	# current_state = 1
	current_state = current_sequence.current_state
	dict_of_current_state = current_sequence.suff_stats["low"][current_state]
	dict_of_state_creation = current_sequence.suff_stats["top"]
	total_num_states = length(keys(dict_of_state_creation)) 
	if sampled_index <= total_num_states
		dict_of_current_state[sampled_index] += 1
		current_sequence.seq_history[time] = sampled_index
		current_sequence.current_state = sampled_index

	elseif sampled_index > total_num_states & sampled_index <= 2 * total_num_states
		if contains(dict_of_current_state, sampled_index - total_num_states)
			dict_of_current_state[sampled_index - total_num_states] += 1
		else
			dict_of_current_state[sampled_index - total_num_states] = 1
		end
		dict_of_state_creation[sampled_index - total_num_states] += 1
		current_sequence.seq_history[time] = sampled_index - total_num_states
		current_sequence.current_state = sampled_index - total_num_states
	else
		dict_of_current_state[total_num_states + 1] = 1
		dict_of_state_creation[total_num_states + 1] = 1
		current_sequence.seq_history[time] = total_num_states + 1
		current_sequence.current_state = total_num_states + 1
	end
	
end


###################RUNNER ##########################


suff_stats = Dict()
suff_stats["low"] = Dict()
suff_stats["top"] = Dict()
suff_stats["low"][1] = Dict()
suff_stats["low"][1][2] = 23
suff_stats["low"][1][3] = 22
suff_stats["top"][1] = 1
suff_stats["top"][2] = 1
suff_stats["top"][3] = 1
seq_history = zeros(SEQUENCE_LENGTH)
current_sequence = sequence(suff_stats, seq_history, 1)

println((createCRFProbVect(current_sequence)))

updateIHMMSuffStat(current_sequence, 2, 2)
println(current_sequence)














