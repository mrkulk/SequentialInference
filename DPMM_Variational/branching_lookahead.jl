#tejasdkulkarni@gmail.com | tejask@mit.edu


function pickNewChildren(current, z_posterior_array_probability, z_posterior_array_cid, PATH_QUEUE, PCNT)
	# Now choose 'p' children and add to queue
	time = current.time
	DEPTH = current.depth

	if ENUMERATION == 1
		normalizing_constant = sum(z_posterior_array_probability)
		z_posterior_array_probability /= normalizing_constant

		for ind=1:length(z_posterior_array_cid)
			child_support = unique(myappend(current.support, z_posterior_array_cid[ind]))
			child_c_aggregate = myappend(current.prev_c_aggregate, z_posterior_array_cid[ind])
			weight = z_posterior_array_probability[ind]
			child = node(unique(child_support), current.weight*weight, DEPTH+1, time+1, child_c_aggregate)
			enqueue!(PATH_QUEUE, child, PCNT)
			PCNT+=1
		end
	else 
		for p=1:INTEGRAL_PATHS
			weight, sampled_cid = sample_from_crp(z_posterior_array_probability, z_posterior_array_cid)
			if sampled_cid == max(current.support)
				child_support = myappend(current.support, sampled_cid+1)
			else
				#indx=findin(current.support, max(current.support))[1] 
				#delete!(current.support, indx)
				child_support = copy(current.support)
			end
			# Adding cluster for each data point until current time for child
			child_c_aggregate = myappend(current.prev_c_aggregate, sampled_cid)
			child = node(unique(child_support), current.weight*weight, DEPTH+1, time+1, child_c_aggregate)
			enqueue!(PATH_QUEUE, child, PCNT)
			PCNT+=1
		end
	end
	return PATH_QUEUE, PCNT
end


function generateCandidateChildren(current_support, time, prev_c_aggregate, N)
	z_posterior_array_probability = []
	z_posterior_array_cid = []

	for j in current_support
		current_c_aggregate = myappend(prev_c_aggregate, j)
		zj_probability = get_posterior_zj(j, current_c_aggregate, time, N, current_support)

		z_posterior_array_probability = myappend(z_posterior_array_probability, zj_probability)
		z_posterior_array_cid = myappend(z_posterior_array_cid, j)
	end
	return z_posterior_array_probability, z_posterior_array_cid
end



function get_weight_lookahead(prev_support, prev_c_aggregate, time, prev_cid, N, lambda_kw)
	
	if LOOKAHEAD_DELTA == 0
		return 1
	end

	PATH_QUEUE = PriorityQueue()
	PCNT = 1

	#time is already t+1
	if prev_cid == max(prev_support)
		t_1_support = unique(myappend(prev_support, prev_cid + 1))
	else
		t_1_support = copy(prev_support)
	end
	z_posterior_array_probability, z_posterior_array_cid = generateCandidateChildren(t_1_support, time, prev_c_aggregate, N)
	current = node(t_1_support, 1, 1, time, prev_c_aggregate, lambda_kw)
	PATH_QUEUE, PCNT = pickNewChildren(current, z_posterior_array_probability, z_posterior_array_cid, PATH_QUEUE, PCNT)

	#Now we propagate t+2 onwards ... 
	while true
		current = dequeue!(PATH_QUEUE)
		if current.depth == LOOKAHEAD_DELTA
			#wARR = []
			#terminate and return with weight
			weight = current.weight
			#wARR = myappend(wARR, weight)
			while length(PATH_QUEUE) > 0 
				elm = dequeue!(PATH_QUEUE)
				if elm.depth != LOOKAHEAD_DELTA
					return weight
				end
				weight += elm.weight
				#wARR = myappend(wARR, elm.weight)
			end
			return weight
		end
		z_posterior_array_probability, z_posterior_array_cid = generateCandidateChildren(current.support, current.time, current.prev_c_aggregate, N)		
		PATH_QUEUE, PCNT = pickNewChildren(current, z_posterior_array_probability, z_posterior_array_cid, PATH_QUEUE, PCNT)
	end
end
