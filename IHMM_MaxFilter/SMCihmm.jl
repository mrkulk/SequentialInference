


using Distributions
using Debug
using PyCall


if ALICE_DATASET_MODE == false
	require("ComputeInferenceError.jl")
end
require("GenerateData.jl")


type sequence
	suff_stats # a dict from "low" and "top" to the suff stats for these levels
	seq_history
	current_state
end
@debug begin
srand(2)
###################PARAMETERS#####################


TOP_ALPHA = 1
LOW_ALPHA = 1
OBS_ALPHA = 1

NUM_PARTICLES = 100
NUM_SAMPLES = 10

obs_sequence = Dict()


# seq_true = []
# for i = 1:length(keys(obs_sequence))
# 	seq_true = vcat(seq_true, obs_sequence[i])
# end


# seq_true = [1.0,1.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,2.0,2.0,1.0,4.0,2.0,2.0,3.0,2.0,2.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,4.0,2.0,2.0,2.0]
# obs = [1.0,1.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,2.0,2.0,1.0,4.0,2.0,2.0,3.0,2.0,2.0,1.0,2.0,2.0,2.0,3.0,3.0,3.0,4.0,2.0,2.0,2.0]

# seq_true = [2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,1.0,1.0,2.0,1.0,1.0]
# obs = [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,1.0,1.0,2.0,3.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,2.0,4.0,1.0,2.0,2.0,1.0,1.0,2.0,1.0,1.0]

# seq_true = [2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,4.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
# obs = [2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,4.0,4.0,1.0,1.0,1.0,1.0,3.0,1.0,1.0,3.0,1.0,2.0,1.0,4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

# seq_true = hidden_state_seq
# obs = observation_seq

# # seq_true = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,2.0,2.0,2.0,1.0,1.0,2.0,2.0,1.0]
# # obs = [3.0,1.0,1.0,1.0,3.0,1.0,2.0,1.0,2.0,2.0,3.0,3.0,2.0,3.0,3.0,3.0,3.0,1.0,3.0,3.0,4.0,3.0,2.0,2.0,2.0,1.0,1.0,2.0,2.0,1.0]

# SEQUENCE_LENGTH = length(seq_true)
# for j = 1:SEQUENCE_LENGTH
# 	obs_sequence[j] = obs[j]
# end
NUM_OBS = 4

###################HELPER FUNCTIONS###############






#################GENERATIVE MODEL#################


function createCRFProbVect(current_sequence, obs)
	current_state = current_sequence.current_state
	dict_of_current_state = current_sequence.suff_stats["low"][current_state]
	dict_of_state_creation = current_sequence.suff_stats["top"]
	#dict_of_obs_suffstat = current_sequence.suff_stats["obs"]
	#current_num_states = length(keys(dict_of_current_state)) #only states generated for the current state
	total_num_states = length(keys(dict_of_state_creation)) #all the states generated
	total_num_transitions = sum(values(dict_of_current_state))
	total_num_top_transition = sum(values(dict_of_state_creation))

	#denom_obs_part1 = OBS_ALPHA * NUM_OBS
	prob_vect = zeros(total_num_states * 2 + 1) #one more for the prob of new state creation
	low_temp_denom = LOW_ALPHA + total_num_transitions
	for i = 1:total_num_states
		try
			prob_vect[i] = dict_of_current_state[i] / low_temp_denom
			#prob_vect[i] *= ((dict_of_obs_suffstat[i][obs] + OBS_ALPHA) / (denom_obs_part1 + sum(values(dict_of_obs_suffstat[i]))) )
		catch
			prob_vect[i] = 0
		end
	end

	top_temp_denom = TOP_ALPHA + total_num_top_transition
	for i = (total_num_states + 1): (total_num_states * 2)
		prob_vect[i] = (LOW_ALPHA / low_temp_denom) * dict_of_state_creation[i - total_num_states] / top_temp_denom
		#prob_vect[i] *= ((dict_of_obs_suffstat[i - total_num_states][obs] + OBS_ALPHA) / (denom_obs_part1 + sum(values(dict_of_obs_suffstat[i - total_num_states]))) )
	end

	prob_vect[length(prob_vect)] = (LOW_ALPHA / low_temp_denom) * (TOP_ALPHA / top_temp_denom) #* (OBS_ALPHA / denom_obs_part1)
	return prob_vect
end




function updateIHMMSuffStat(current_sequence, sampled_index, time, obs) #use sampled index to find if it's new or old state 
	#sample index is for the 'time' and current_sequence is up to 'time - 1'
	if time == 1
		current_state = -1
	else
		current_state = current_sequence.current_state
	end
	denom_obs_part1 = OBS_ALPHA * NUM_OBS
	dict_of_current_state = current_sequence.suff_stats["low"][current_state]
	dict_of_state_creation = current_sequence.suff_stats["top"]
	dict_of_obs_suffstat = current_sequence.suff_stats["obs"]

	total_num_states = length(keys(dict_of_state_creation)) 
	if sampled_index <= total_num_states
		if contains(keys(dict_of_current_state), sampled_index)
			dict_of_current_state[sampled_index] += 1
		else
			dict_of_current_state[sampled_index] = 1
		end
		current_sequence.seq_history[time] = sampled_index
		current_sequence.current_state = sampled_index
		dict_of_obs_suffstat[sampled_index][obs] += 1
		weight = ((dict_of_obs_suffstat[sampled_index][obs] + OBS_ALPHA) / (denom_obs_part1 + sum(values(dict_of_obs_suffstat[sampled_index]))) )

	elseif sampled_index > total_num_states && sampled_index <= 2 * total_num_states
		if contains(keys(dict_of_current_state), sampled_index - total_num_states)
			dict_of_current_state[sampled_index - total_num_states] += 1
		else
			dict_of_current_state[sampled_index - total_num_states] = 1
		end
		if contains(keys(dict_of_state_creation), sampled_index - total_num_states)
			dict_of_state_creation[sampled_index - total_num_states] += 1
		else
			dict_of_state_creation[sampled_index - total_num_states] = 1
		end
		current_sequence.seq_history[time] = sampled_index - total_num_states
		current_sequence.current_state = sampled_index - total_num_states
		dict_of_obs_suffstat[sampled_index - total_num_states][obs] += 1
		weight = ((dict_of_obs_suffstat[sampled_index - total_num_states][obs] + OBS_ALPHA) / 
			(denom_obs_part1 + sum(values(dict_of_obs_suffstat[sampled_index - total_num_states]))) )

	else
		dict_of_current_state[total_num_states + 1] = 1
		dict_of_state_creation[total_num_states + 1] = 1
		
		current_sequence.suff_stats["low"][total_num_states + 1] = Dict()
		current_sequence.seq_history[time] = total_num_states + 1
		current_sequence.current_state = total_num_states + 1
		dict_of_obs_suffstat[total_num_states + 1] = Dict()
		for obs_key = 1:NUM_OBS
				dict_of_obs_suffstat[total_num_states + 1][obs_key] = 0
		end
		dict_of_obs_suffstat[total_num_states + 1][obs] += 1
		weight = (OBS_ALPHA / denom_obs_part1)
	end

	return weight
	
end


###################RUNNER ##########################



function mainSMC(seed)
	#function body
data_dict = main_generate(seed)
seq_true = data_dict["hid"]
obs = data_dict["obs"]

SEQUENCE_LENGTH = length(seq_true)
for j = 1:SEQUENCE_LENGTH
	obs_sequence[j] = obs[j]
end

#########INIT SEQ
suff_stats = Dict()
suff_stats["low"] = Dict()
suff_stats["top"] = Dict()
suff_stats["low"][-1] = Dict()
#suff_stats["low"][-1][1] = 0
suff_stats["top"] = Dict()
#suff_stats["top"][1] = 0
suff_stats["obs"] = Dict()
suff_stats["obs"][1] = Dict()
for obs_key = 1:NUM_OBS
	suff_stats["obs"][1][obs_key] = 0
end

seq_history = zeros(SEQUENCE_LENGTH)
current_sequence = sequence(suff_stats, seq_history, -1)



###################INIT PARTICLES
particle_dict = Dict()
weight_vect = zeros(NUM_PARTICLES)

for p = 1:NUM_PARTICLES
	particle_dict[p] = deepcopy(current_sequence)
end



for t = 1:SEQUENCE_LENGTH
	particle_dict_temp = Dict()
	for p_num = 1:NUM_PARTICLES
		prob_vect = createCRFProbVect(particle_dict[p_num], obs_sequence[t])
		# println("probVect: ", prob_vect)

		prob_vect /= sum(prob_vect)
		sample_arr = rand(Multinomial(1,prob_vect))
		idx = findin(sample_arr, 1)[1]
		# println("indx: ", idx)
		# println()
		weight = updateIHMMSuffStat(particle_dict[p_num], idx, t, obs_sequence[t])
		weight_vect[p_num] = weight
	end
	if t != SEQUENCE_LENGTH
		particle_dict_temp = deepcopy(particle_dict)
		normalized_weight_vect = weight_vect / sum(weight_vect)
		for j = 1:NUM_PARTICLES
			sample_arr = rand(Multinomial(1, normalized_weight_vect))
			idx = findin(sample_arr, 1)[1]
			particle_dict[j] = deepcopy(particle_dict_temp[idx])
		end
	end

end



normalized_weight_vect = weight_vect / sum(weight_vect)

total_error = 0
for h = 1:NUM_SAMPLES
	sample_arr = rand(Multinomial(1, normalized_weight_vect))
	idx = findin(sample_arr, 1)[1]
	seq_inferred = particle_dict[idx].seq_history
	total_error += computeError(seq_inferred, seq_true)
end
error = total_error / NUM_SAMPLES
SMC_error = error
println("SMC error: ", error)
return SMC_error
end 

end











