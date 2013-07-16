#Variational Lookahead 
#Ardavan Saeedi & Tejas Kulkarni


function get_weight_lookahead(prev_weight, prev_support, prev_c_aggregate, time, prev_cid, N, prev_lambda_kw)
	VARIATIONAL_ITERATIONS
	for iter=1:VARIATIONAL_ITERATIONS
		for t=time:time+LOOKAHEAD_DELTA

		end
	end
end
"""	
	if LOOKAHEAD_DELTA == 0 || LOOKAHEAD_DELTA == 1
		return prev_weight
	end

	PATH_QUEUE = PriorityQueue()
	PCNT = 1

	#time is already t+1
	if prev_cid == max(prev_support)
		t_1_support = unique(myappend(prev_support, prev_cid + 1))
	else
		t_1_support = deepcopy(prev_support)
	end

	println("====================[LAMBDA FROM TOP LEVEL]====================")
	println(prev_lambda_kw)
	println("====================[[get_weight_lookahead time:", time ," prev_cid: ", prev_cid ,"]]====================")

	z_posterior_array_probability, z_posterior_array_cid, lambda_kw_arr = generateCandidateChildren(t_1_support, time, prev_c_aggregate, N, prev_lambda_kw)
	current = node(t_1_support, prev_weight, 1, time, prev_c_aggregate, prev_lambda_kw)

	PATH_QUEUE, PCNT = pickNewChildren(current, z_posterior_array_probability, z_posterior_array_cid, lambda_kw_arr, PATH_QUEUE, PCNT)


	#Now we propagate t+2 onwards ... 
	while true
		current = dequeue!(PATH_QUEUE)
		if current.depth == LOOKAHEAD_DELTA
			wARR = []
			#terminate and return with weight
			#weight = exp(current.weight)
			wARR = myappend(wARR, current.weight)
			while length(PATH_QUEUE) > 0 
				elm = dequeue!(PATH_QUEUE)
				if elm.depth != LOOKAHEAD_DELTA
					#return log(weight)
					break
				end
				#weight += exp(elm.weight)
				wARR = myappend(wARR, elm.weight)
			end
			#return log(weight)
			#println("ESCAPE: ", wARR)
			return logsumexp(wARR)
		end
		z_posterior_array_probability, z_posterior_array_cid, lambda_kw_arr = generateCandidateChildren(current.support, current.time, current.prev_c_aggregate, N, current.prev_lambda_kw)		
		PATH_QUEUE, PCNT = pickNewChildren(current, z_posterior_array_probability, z_posterior_array_cid, lambda_kw_arr, PATH_QUEUE, PCNT)
		#println("*********")
	end
end
"""
