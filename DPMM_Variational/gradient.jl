
using Debug

@debug begin 

function gradient_v(sampled_cid, time, N, is_new_cid)
	if is_new_cid == false  #existing cluster
		prev_soft_v = deepcopy(particles[time-1][N]["hidden_state"]["soft_v"])
	end

	alpha = hyperparameters["a"]
	v_sufficient_stats = 0

	if is_new_cid == true
		#creation of new cluster
		particles[time][N]["hidden_state"]["soft_v"][sampled_cid] =  LRATE*(alpha)
	else
		particles[time][N]["hidden_state"]["soft_v"][sampled_cid] = prev_soft_v[sampled_cid] + LRATE*(-prev_soft_v[sampled_cid] + alpha + NUM_DOCS*v_sufficient_stats)
	end
end


function gradient_soft_lambda_u(sampled_cid, wordArr, posterior, time, N, is_new_cid)
	eta = hyperparameters["eta"]
	if is_new_cid == false  #existing cluster
		prev_soft_lambda_kw = deepcopy(particles[time-1][N]["hidden_state"]["soft_lambda"])
		prev_soft_u = deepcopy(particles[time-1][N]["hidden_state"]["soft_u"])
	end
	posterior = 1 ##HARD ASSIGNMENT
	particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid] = Dict()
	particles[time][N]["hidden_state"]["soft_u"][sampled_cid] = Dict()
	for word = 1:V
		if is_new_cid == true #existing cluster
			particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid][word] =  LRATE*(eta)
		else
			particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid][word] = prev_soft_lambda_kw[sampled_cid][word] + LRATE*(-prev_soft_lambda_kw[sampled_cid][word] + eta + NUM_DOCS*(posterior*wordArr[word]))
		end			
	end

	if is_new_cid == true
		particles[time][N]["hidden_state"]["soft_u"][sampled_cid] = LRATE*(1)
	else
		particles[time][N]["hidden_state"]["soft_u"][sampled_cid] = prev_soft_u[sampled_cid] + LRATE*(-prev_soft_u[sampled_cid] + 1 + NUM_DOCS*posterior)
	end	
end



function update_newcluster_statistics(sampled_cid, data, time, wordArr, posterior, N)
	## create new lambda ##
	particles[time][N]["hidden_state"]["lambda"][sampled_cid] = Dict()
	for word = 1:V
		particles[time][N]["hidden_state"]["lambda"][sampled_cid][word] = hyperparameters["eta"] + wordArr[word]
	end
	gradient_soft_lambda_u( sampled_cid, wordArr, posterior, time, N, true)
 	gradient_v(sampled_cid, time, N, true)
end



function update_existingcluster_statistics(sampled_cid, data, time, wordArr, posterior, N, lambda_statistics)
	for word=1:V
		particles[time][N]["hidden_state"]["lambda"][sampled_cid][word] = lambda_statistics[word]
	end
	gradient_soft_lambda_u(sampled_cid, wordArr, posterior, time, N, false)
	gradient_v(sampled_cid, time, N, false)
end


function update_all_not_chosen_ks(sampled_cid, support, time, N)
	prev_soft_lambda = deepcopy(particles[time-1][N]["hidden_state"]["soft_lambda"])
	prev_soft_u = deepcopy(particles[time-1][N]["hidden_state"]["soft_u"])
	prev_soft_v = deepcopy(particles[time-1][N]["hidden_state"]["soft_v"])
	eta = hyperparameters["eta"]; a = hyperparameters["a"]

	for cid in support
		if cid != sampled_cid
			if haskey(particles[time][N]["hidden_state"]["soft_lambda"], cid) == false ###create if not existent
				if cid != max(support)
					throw("ERROR IN [update_all_not_chosen_ks]")
				end
				particles[time][N]["hidden_state"]["soft_lambda"][cid] = Dict()
				
				for word=1:V
					particles[time][N]["hidden_state"]["soft_lambda"][cid][word] =  LRATE*(eta)
				end
				particles[time][N]["hidden_state"]["soft_u"][cid] = LRATE*(1)
				if sampled_cid > cid
					v_sufficient_stats = 1
				else
					v_sufficient_stats = 0
				end
				particles[time][N]["hidden_state"]["soft_v"][cid] = LRATE*(a + NUM_DOCS*v_sufficient_stats)

			else
				for word=1:V
					particles[time][N]["hidden_state"]["soft_lambda"][cid][word] = prev_soft_lambda[cid][word] + LRATE*(-prev_soft_lambda[cid][word] + eta)
				end
				particles[time][N]["hidden_state"]["soft_u"][cid] = prev_soft_u[cid] + LRATE*(-prev_soft_u[cid] + 1)
				if sampled_cid > cid
					v_sufficient_stats = 1
				else
					v_sufficient_stats = 0
				end
				particles[time][N]["hidden_state"]["soft_v"][cid] = prev_soft_v[cid] + LRATE*(-prev_soft_v[cid] + a + NUM_DOCS*v_sufficient_stats)
			end

		end
	end
end



end


