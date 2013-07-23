
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


function stratifiedMaxFiltering(particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, NUM_PARTICLES)
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	maxfilter_cid_array = maxfilter_cid_array[perm]
	maxfilter_particle_struct = maxfilter_particle_struct[perm]
	#println(maxfilter_cid_array)
	i=1
	
	maxfilter_cid_array = unique(maxfilter_cid_array)
	unique_total_cids = length(maxfilter_cid_array)
	cnt=1
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
	end
end 


end