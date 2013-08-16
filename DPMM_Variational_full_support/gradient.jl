
using Debug

@debug begin 

function gradient_v(sampled_cid, time, N, is_new_cid)
	alpha = hyperparameters["a"]
	v_sufficient_stats = 0

	if is_new_cid == true
		#creation of new cluster
		particles[time][N]["hidden_state"]["soft_v"][sampled_cid] =  LRATE*(alpha)
	else
		prev_soft_v = particles[time-1][N]["hidden_state"]["soft_v"]	
		particles[time][N]["hidden_state"]["soft_v"][sampled_cid] = prev_soft_v[sampled_cid] + LRATE*(-prev_soft_v[sampled_cid] + alpha + NUM_DOCS*v_sufficient_stats)
	end
end


function gradient_soft_lambda_u(sampled_cid, wordArr, posterior, time, N, is_new_cid)
	eta = hyperparameters["eta"]
	posterior = 1

	if is_new_cid == true
		particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid] = Dict()
		for word=1:V
			particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid][word] =  LRATE*(eta + NUM_DOCS*wordArr[word])
		end
		particles[time][N]["hidden_state"]["soft_u"][sampled_cid] = LRATE*(1 + NUM_DOCS)
	else
		prev_soft_lambda_kw = particles[time-1][N]["hidden_state"]["soft_lambda"]
		prev_soft_u = particles[time-1][N]["hidden_state"]["soft_u"]
		particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid] = Dict()
		for word=1:V
			particles[time][N]["hidden_state"]["soft_lambda"][sampled_cid][word] = prev_soft_lambda_kw[sampled_cid][word] + LRATE*(-prev_soft_lambda_kw[sampled_cid][word] + eta + NUM_DOCS*(posterior*wordArr[word]))
		end
		particles[time][N]["hidden_state"]["soft_u"][sampled_cid] = prev_soft_u[sampled_cid] + LRATE*(-prev_soft_u[sampled_cid] + 1 + NUM_DOCS*posterior)		
	end
end



function update_newcluster_statistics(sampled_cid, data, time, wordArr, posterior, N)
	## create new lambda ##

	particles[time][N]["hidden_state"]["lambda"][sampled_cid] = Dict()
	for word = 1:V
		particles[time][N]["hidden_state"]["lambda"][sampled_cid][word] = hyperparameters["eta"] + wordArr[word]
	end
	#gradient_soft_lambda_u( sampled_cid, wordArr, posterior, time, N, true)
 	#gradient_v(sampled_cid, time, N, true)
end



function update_existingcluster_statistics(sampled_cid, data, time, wordArr, posterior, N, lambda_statistics)
	for word=1:V
		particles[time][N]["hidden_state"]["lambda"][sampled_cid][word] = lambda_statistics[word]
	end
	#gradient_soft_lambda_u(sampled_cid, wordArr, posterior, time, N, false)
	#gradient_v(sampled_cid, time, N, false)
end


function update_all_not_chosen_ks(sampled_cid, support, time, N, max_root_support)
	eta = hyperparameters["eta"]; a = hyperparameters["a"]
	prev_soft_lambda = particles[time-1][N]["hidden_state"]["soft_lambda"]
	prev_soft_u = particles[time-1][N]["hidden_state"]["soft_u"]
	prev_soft_v = particles[time-1][N]["hidden_state"]["soft_v"]

	for cid in support
		if cid != sampled_cid && cid < max_root_support
		
			is_new_cid = (has(particles[time][N]["hidden_state"]["soft_lambda"], cid) == false)

			if is_new_cid == true
				particles[time][N]["hidden_state"]["soft_lambda"][cid] = Dict()
				prev_soft_u[cid] = 0
				prev_soft_v[cid] = 0
			end

			for word=1:V
				if is_new_cid == true
					particles[time][N]["hidden_state"]["soft_lambda"][cid][word] = LRATE*(eta)
				else
					particles[time][N]["hidden_state"]["soft_lambda"][cid][word] = prev_soft_lambda[cid][word] + LRATE*(-prev_soft_lambda[cid][word] + eta)
				end
			end

			particles[time][N]["hidden_state"]["soft_u"][cid]= prev_soft_u[cid] + LRATE*(-prev_soft_u[cid] + 1)
			particles[time][N]["hidden_state"]["soft_v"][cid]= prev_soft_v[cid] + LRATE*(-prev_soft_v[cid] + a)

		end
	end
end



end


