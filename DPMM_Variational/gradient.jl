
using Debug

@debug begin 

function gradient_v(prev_soft_v, cid, EXP_z_posterior_array_probability, z_posterior_array_cid, LRATE, alpha, NUM_DOCS, time, N, is_new_cid)

	sufficient_stats = 0
	for i=1:length(z_posterior_array_cid)
		if z_posterior_array_cid[i] > cid
			sufficient_stats += EXP_z_posterior_array_probability[i]
		end
	end

	if is_new_cid == false
		particles[time][N]["hidden_state"]["cache"]["soft_v"][cid] = prev_soft_v[cid] + LRATE*(-prev_soft_v[cid] + alpha + NUM_DOCS*sufficient_stats)
	else
		particles[time][N]["hidden_state"]["cache"]["soft_v"][cid] =  LRATE*(alpha + NUM_DOCS*sufficient_stats)
	end
end


function gradient_soft_lambda_u(cid, document, wordArr, posterior, time, N, eta, is_new_cid)
	if is_new_cid == false  #existing cluster
		prev_soft_lambda_kw = particles[time-1][N]["hidden_state"]["soft_lambda"]
		prev_soft_u = particles[time-1][N]["hidden_state"]["soft_u"]
	end
	posterior = 1 ##HARD ASSIGNMENT
	particles[time][N]["hidden_state"]["cache"]["soft_lambda"][cid] = Dict()
	particles[time][N]["hidden_state"]["cache"]["soft_u"][cid] = Dict()
	for word = 1:V
		if is_new_cid == false #existing cluster
			particles[time][N]["hidden_state"]["cache"]["soft_lambda"][cid][word] = prev_soft_lambda_kw[cid][word] + LRATE*(-prev_soft_lambda_kw[cid][word] + eta + NUM_DOCS*(posterior*wordArr[word]))
		else
			particles[time][N]["hidden_state"]["cache"]["soft_lambda"][cid][word] =  LRATE*(eta + NUM_DOCS*(posterior*wordArr[word]))
		end			
	end

	if is_new_cid == false
		particles[time][N]["hidden_state"]["cache"]["soft_u"][cid] = prev_soft_u[cid] + LRATE*(-prev_soft_u[cid] + 1 + NUM_DOCS*posterior)
	else
		particles[time][N]["hidden_state"]["cache"]["soft_u"][cid] = LRATE*(1 + NUM_DOCS*posterior)
	end	
end



end


