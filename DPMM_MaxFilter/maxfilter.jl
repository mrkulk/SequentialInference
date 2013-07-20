
using Debug
@debug begin 

#Tejas D K : tejask@mit.edu
function maxFilter(particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, NUM_PARTICLES)
	#Algorithm proposed by Sam Gershman
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	maxfilter_cid_array = maxfilter_cid_array[perm]
	#println(maxfilter_cid_array)
	for i=1:NUM_PARTICLES
		state=Dict()
		sampled_cid = maxfilter_cid_array[i]
		state["c"] = sampled_cid
		state["c_aggregate"] = myappend(particles_t_minus_1[i]["hidden_state"]["c_aggregate"], sampled_cid)
		particles_t[i]["hidden_state"]=state
	end
end 


function stratifiedMaxFiltering(particles_t, particles_t_minus_1, maxfilter_probability_array, maxfilter_cid_array, NUM_PARTICLES)
	perm = sortperm(maxfilter_probability_array, Sort.Reverse)
	maxfilter_cid_array = maxfilter_cid_array[perm]
	#println(maxfilter_cid_array)
	i=1
	filledList = []
	unique_total_cids = length(unique(maxfilter_cid_array))
	cnt=1
	unique_total_cids = min(unique_total_cids, NUM_PARTICLES)
	while i <= unique_total_cids
		state=Dict()
		sampled_cid = maxfilter_cid_array[cnt]
		cnt+=1
		if length(findin(filledList, sampled_cid)) == 0
			filledList = myappend(filledList, sampled_cid)
			state["c"] = sampled_cid
			state["c_aggregate"] = myappend(particles_t_minus_1[i]["hidden_state"]["c_aggregate"], sampled_cid)
			particles_t[i]["hidden_state"]=state
			print("<",i,">")
			i+=1
		end
	end

	if unique_total_cids < NUM_PARTICLES
		new_num = abs(NUM_PARTICLES - unique_total_cids)
		println(new_num, " ", unique_total_cids)
		for j=i:i+new_num-1
			sampled_cid = maxfilter_cid_array[j]
			filledList = myappend(filledList, sampled_cid)
			state["c"] = sampled_cid
			state["c_aggregate"] = myappend(particles_t_minus_1[j]["hidden_state"]["c_aggregate"], sampled_cid)
			particles_t[j]["hidden_state"]=state
			print("<",j,">")
		end
	end

	println(length(particles_t))
end 


end