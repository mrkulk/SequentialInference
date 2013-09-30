
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


function stratifiedMaxFiltering(time, particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, NUM_PARTICLES)

	## BAD [buy why?]logperm = sortperm(log_maxfilter_probability_array, Sort.Reverse)
	#_maxfilter_cid_array = maxfilter_cid_array[logperm]
	#_maxfilter_particle_struct = maxfilter_particle_struct[logperm]

	# normalized_maxfilter_prob_array = maxfilter_probability_array/sum(maxfilter_probability_array)
	# prev_normalized_prob_array = Float64[ float(particles_t_minus_1[pid]["weight"]) for pid in [1:NUM_PARTICLES] ]
	# @bp
	# maxfilter_probability_array = normalized_maxfilter_prob_array .* prev_normalized_prob_array

	# perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	
	maxfilter_probability_array = maxfilter_probability_array/sum(maxfilter_probability_array)
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)

	#println( sum([i<0 for i in log_maxfilter_probability_array]) - length(log_maxfilter_probability_array))
	#println(log_maxfilter_probability_array)
	#if perm != logperm
	#	@bp
	#end


	maxfilter_cid_array = maxfilter_cid_array[perm]
	maxfilter_particle_struct = maxfilter_particle_struct[perm]
	maxfilter_probability_array = maxfilter_probability_array[perm]

	state=Dict()
	state["c"] = maxfilter_cid_array[1]
	state["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[1]]["hidden_state"]["c_aggregate"], state["c"])
	particles_t[1]["hidden_state"]=state	
	particles_t[1]["weight"] = maxfilter_probability_array[1]

	particle_cnt = 2
	unique_indices = []
	
	for i=2:length(maxfilter_cid_array)
		last = maxfilter_cid_array[i-1]
		cur =  maxfilter_cid_array[i]
		if cur != last
			if particle_cnt > NUM_PARTICLES
				break
			else
				unique_indices = myappend(unique_indices, i)
				state=Dict()
				state["c"] = maxfilter_cid_array[i]
				state["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[i]]["hidden_state"]["c_aggregate"], state["c"])
				particles_t[particle_cnt]["hidden_state"]=state
				particles_t[particle_cnt]["weight"] = maxfilter_probability_array[i]
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
			state["c"] = maxfilter_cid_array[indx]
			state["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[indx]]["hidden_state"]["c_aggregate"], state["c"])
			particles_t[p]["hidden_state"]=state
			particles_t[particle_cnt]["weight"] = maxfilter_probability_array[indx]
		end
	end

	#println(length(particles_t))
	"""	
	unique_maxfilter_cid_array = unique(maxfilter_cid_array)
	unique_total_cids = length(unique_maxfilter_cid_array)
	unique_total_cids = min(unique_total_cids, NUM_PARTICLES)

	END = NaN
	if unique_total_cids >= NUM_PARTICLES
		END = NUM_PARTICLES
	end
	if NUM_PARTICLES > unique_total_cids
		END = unique_total_cids
	end

	for i=1:END
		state=Dict()
		state["c"] = maxfilter_cid_array[i]

		state["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[i]]["hidden_state"]["c_aggregate"], state["c"])
		particles_t[i]["hidden_state"]=state	
		#println(state["c_aggregate"])	
		#print("(",maxfilter_particle_struct[i], ") ")
	end

	IND_CNT = 1
	if NUM_PARTICLES > unique_total_cids
		for i=END+1:NUM_PARTICLES
			state=Dict()
			#state["c"] = maxfilter_cid_array[(IND_CNT%unique_total_cids)+1]#[randi(unique_total_cids)] ##random vs deterministic stratified
			state["c"] = maxfilter_cid_array[randi(unique_total_cids)] ##random vs deterministic stratified
			IND_CNT +=1
			state["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[i]]["hidden_state"]["c_aggregate"], state["c"])
			particles_t[i]["hidden_state"]=state
			#println(state["c_aggregate"])
			#print("(",maxfilter_particle_struct[i], ") ")
		end
	end"""


end 


end