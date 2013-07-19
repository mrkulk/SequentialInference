
#Tejas D K : tejask@mit.edu

function maxFilter(particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, NUM_PARTICLES)
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	maxfilter_cid_array = maxfilter_cid_array[perm]
	println(maxfilter_cid_array)
	for i=1:NUM_PARTICLES
		state=Dict()
		sampled_cid = maxfilter_cid_array[i]
		state["c"] = sampled_cid
		state["c_aggregate"] = myappend(particles_t_minus_1[i]["hidden_state"]["c_aggregate"], sampled_cid)
		particles_t[i]["hidden_state"]=state
	end
end 