
using Debug
@debug begin 

#Tejas D K : tejask@mit.edu
function maxFilter(particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, NUM_PARTICLES)
	#Algorithm proposed by Sam Gershman
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	maxfilter_cid_array = maxfilter_cid_array[perm]
	maxfilter_particle_struct = maxfilter_particle_struct[perm]

	#println(maxfilter_cid_array)
	for i=1:NUM_PARTICLES
		state=Dict()
		sampled_cid = maxfilter_cid_array[i]
		state["c"] = sampled_cid
		state["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[i]]["hidden_state"]["c_aggregate"], sampled_cid)
		particles_t[i]["hidden_state"]=state
	end
end 


function stratifiedMaxFiltering(time, particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, NUM_PARTICLES, log_maxfilter_probability_array, support_array_putative)


	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	hash = Dict()
	record_indx = []
	for i = 1:length(perm)
		if haskey(hash, maxfilter_cid_array[perm[i]]) == false
			hash[maxfilter_cid_array[perm[i]]] = perm[i]
			record_indx = myappend(record_indx, perm[i])
		end
	end

	len_rind = length(record_indx)
	pind=1

	for i = 1:NUM_PARTICLES
		if pind > len_rind
			pind=1
		end
		sampled_indx = record_indx[pind]
		pind+=1

		particles_t[i]["hidden_state"]["c"] = maxfilter_cid_array[sampled_indx]
		particles_t[i]["hidden_state"]["sampled_indx"] = sampled_indx
		particles_t[i]["hidden_state"]["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["c_aggregate"], particles_t[i]["hidden_state"]["c"])
		particles_t[i]["weight"] = log_maxfilter_probability_array[sampled_indx]
		particles_t[i]["max_support"] = support_array_putative[sampled_indx] #######################################################

		particles_t[i]["hidden_state"]["lambda"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["lambda"])
		particles_t[i]["hidden_state"]["soft_lambda"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["soft_lambda"])
		particles_t[i]["hidden_state"]["soft_u"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["soft_u"])
		particles_t[i]["hidden_state"]["soft_v"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["soft_v"])
	end

"""
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	maxfilter_cid_array = maxfilter_cid_array[perm]
	maxfilter_particle_struct = maxfilter_particle_struct[perm]
	log_maxfilter_probability_array = log_maxfilter_probability_array[perm]
	support_array_putative = support_array_putative[perm]

	particles_t[1]["hidden_state"]["c"] = maxfilter_cid_array[1]
	particles_t[1]["hidden_state"]["sampled_indx"] = perm[1] #######################################################
	particles_t[1]["hidden_state"]["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[1]]["hidden_state"]["c_aggregate"], particles_t[1]["hidden_state"]["c"])

	particles_t[1]["weight"] = log_maxfilter_probability_array[1] #######################################################
	particles_t[1]["max_support"] = support_array_putative[1] #######################################################

	particle_cnt = 2
	unique_indices = []
	perm_index = []

	for i=2:length(maxfilter_cid_array)
		last = maxfilter_cid_array[i-1]
		cur =  maxfilter_cid_array[i]
		if cur != last
			if particle_cnt > NUM_PARTICLES
				break
			else
				unique_indices = myappend(unique_indices, i)
				perm_index = myappend(perm_index, perm[i])
				particles_t[particle_cnt]["hidden_state"]["c"] = maxfilter_cid_array[i]
				particles_t[particle_cnt]["hidden_state"]["sampled_indx"] = perm[i] #######################################################
				particles_t[particle_cnt]["hidden_state"]["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[i]]["hidden_state"]["c_aggregate"], particles_t[particle_cnt]["hidden_state"]["c"])

				particles_t[particle_cnt]["weight"] = log_maxfilter_probability_array[i]  #######################################################
				particles_t[particle_cnt]["max_support"] = support_array_putative[i] #######################################################
				particle_cnt+=1
			end
		end
		i+=1
	end

	if NUM_PARTICLES >= particle_cnt
		len_unique_indices = length(unique_indices)
		for p=particle_cnt:NUM_PARTICLES
			state=Dict()
			#indx = unique_indices[p%len_unique_indices + 1]
			#indx = unique_indices[randi(len_unique_indices)]
			indx = unique_indices[1]
			perm_indx = perm_index[1]

			particles_t[p]["hidden_state"]["c"] = maxfilter_cid_array[indx]
			particles_t[p]["hidden_state"]["sampled_indx"] = perm_indx #######################################################
			particles_t[p]["hidden_state"]["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[indx]]["hidden_state"]["c_aggregate"], particles_t[p]["hidden_state"]["c"])

			particles_t[p]["weight"] = log_maxfilter_probability_array[indx]  #######################################################
			particles_t[p]["max_support"] = support_array_putative[indx] #######################################################
		end
	end

"""
end 


end