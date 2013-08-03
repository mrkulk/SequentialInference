using Distributions
using Debug
using PyCall





type sequence
	suff_stats # a dict from "low" and "top" to the suff stats for these levels
	seq_history
	current_state
end
@debug begin
srand(1)
###################PARAMETERS#####################


TOP_ALPHA = 1
LOW_ALPHA = 1
OBS_ALPHA = 1
NUM_OBS = 4
SEQUENCE_LENGTH = 3
NUM_PARTICLES = 5


obs_sequence = Dict()
obs_sequence[1] = 1
obs_sequence[2] = 1
obs_sequence[3] = 2






###################HELPER FUNCTIONS###############






#################GENERATIVE MODEL#################


function createCRFProbVect(current_sequence, obs)
	current_state = current_sequence.current_state
	dict_of_current_state = current_sequence.suff_stats["low"][current_state]
	dict_of_state_creation = current_sequence.suff_stats["top"]
	dict_of_obs_suffstat = current_sequence.suff_stats["obs"]
	#current_num_states = length(keys(dict_of_current_state)) #only states generated for the current state
	total_num_states = length(keys(dict_of_state_creation)) #all the states generated
	total_num_transitions = sum(values(dict_of_current_state))
	total_num_top_transition = sum(values(dict_of_state_creation))

	denom_obs_part1 = OBS_ALPHA * NUM_OBS
	prob_vect = zeros(total_num_states * 2 + 1) #one more for the prob of new state creation
	low_temp_denom = LOW_ALPHA + total_num_transitions
	for i = 1:total_num_states
		try
			prob_vect[i] = dict_of_current_state[i] / low_temp_denom
			prob_vect[i] *= ((dict_of_obs_suffstat[i][obs] + OBS_ALPHA) / (denom_obs_part1 + sum(values(dict_of_obs_suffstat[i]))) )
		catch
			prob_vect[i] = 0
		end
	end

	top_temp_denom = TOP_ALPHA + total_num_top_transition
	for i = (total_num_states + 1): (total_num_states * 2)
		prob_vect[i] = (LOW_ALPHA / low_temp_denom) * dict_of_state_creation[i - total_num_states] / top_temp_denom
		prob_vect[i] *= ((dict_of_obs_suffstat[i - total_num_states][obs] + OBS_ALPHA) / (denom_obs_part1 + sum(values(dict_of_obs_suffstat[i - total_num_states]))) )
	end

	prob_vect[length(prob_vect)] = (LOW_ALPHA / low_temp_denom) * (TOP_ALPHA / top_temp_denom) * (OBS_ALPHA / denom_obs_part1)
	return prob_vect
end




function updateIHMMSuffStat(current_sequence, sampled_index, time, obs) #use sampled index to find if it's new or old state 
	#sample index is for the 'time' and current_sequence is up to 'time - 1'
	if time == 1
		current_state = -1
	else
		current_state = current_sequence.current_state
	end

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
	end
	
end


###################RUNNER ##########################





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
for p = 1:NUM_PARTICLES
	particle_dict[p] = deepcopy(current_sequence)
end



for t = 1:SEQUENCE_LENGTH
	
	prob_vect_concated = []
	state_particle_pair_list = []
	#creating the prob vectors for time t and sorting the probs
	for p_num = 1:NUM_PARTICLES
		prob_vect = createCRFProbVect(particle_dict[p_num], obs_sequence[t])
		prob_vect_concated = vcat(prob_vect_concated, prob_vect)
		for ll = 1:length(prob_vect)
			state_particle_pair_list = vcat(state_particle_pair_list, (ll, p_num) )
		end
	end
	perm = sortperm(prob_vect_concated, Sort.Reverse)
	prob_vect_concated = prob_vect_concated[perm]
	state_particle_pair_list = state_particle_pair_list[perm]
	list_of_states = [pair[1] for pair in state_particle_pair_list]
	list_of_particles = [pair[2] for pair in state_particle_pair_list]
	unique_state_list = unique(list_of_states)

	
	

	#now extending the particles for time t by selecting from the particles (note that we are still at time t)
	particle_dict_temp = Dict()
	if t == 2
		@bp
	end
	num_of_uniques_added = 0 
	for lll = 1:min(length(unique_state_list), NUM_PARTICLES)
		first_index = findin(list_of_states, unique_state_list[lll])[1]
		max_particle = state_particle_pair_list[first_index][2] #finding the cluster with max prob
		prob_of_index = prob_vect_concated[first_index]#check if it's not zero
		if prob_of_index != 0
			particle_dict_temp[lll] = deepcopy(particle_dict[max_particle])
			updateIHMMSuffStat(particle_dict_temp[lll], list_of_states[first_index], t, obs_sequence[t])
			num_of_uniques_added += 1
		end

	end
	
	if num_of_uniques_added < NUM_PARTICLES
		nonunique_particles = (num_of_uniques_added + 1):NUM_PARTICLES
		# for kk in state_particle_pair_list
		# 	if contains(unique_state_list, kk[1]) == false
		# 		vcat(nonunique_particles, kk[2])
		# 	end
		# end
		
		for s in nonunique_particles
			particle_dict_temp[s] = deepcopy(particle_dict_temp[1])
			
		end
	end
	particle_dict = deepcopy(particle_dict_temp)
	println(particle_dict)
	println()
end


# println((createCRFProbVect(current_sequence, 1)))

# updateIHMMSuffStat(current_sequence, 1, 1, 1)
# println(current_sequence)
# println((createCRFProbVect(current_sequence, 1)))


end


########################################################################
# suff_stats["low"][1][2] = 23
# suff_stats["low"][1][3] = 22
# suff_stats["top"][1] = 1
# suff_stats["top"][2] = 1
# suff_stats["top"][3] = 1








