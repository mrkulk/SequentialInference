#Variational Lookahead 
#Ardavan Saeedi & Tejas Kulkarni
using Debug
@debug begin 


function sample_q_variational(cid, current_support, time, N, max_support_value, soft_lambda, soft_u,soft_v, data)
	eta = hyperparameters["eta"]
	posterior = 0

	if cid == max_support_value #new cluster
		numerator1 = lgamma(eta*V)
		denominator1 = V*lgamma(eta)
		posterior += numerator1 - denominator1 
		numerator2 = 0;
		wordArr = zeros(V)
		words_in_this_doc = collect(values(data[time]))
		for word = 1:V
			indices = findin(words_in_this_doc, word)
			tmp=length(indices)
			wordArr[word] = tmp
			numerator2 += lgamma(eta + tmp)
		end
		denominator2 = lgamma(eta*V + length(data[time]))
		
		posterior += numerator2		
		posterior -= denominator2

		for k in current_support
			posterior += log(soft_v[k]) - log(soft_v[k] + soft_u[k])
		end

	else #existing cluster
		numerator1 = 0; tmp_denominator1 = 0; #this is first side page 5 from Chong et al
		numerator2 = 0; tmp_denominator2 = 0; #this is second side page 5 from Chong et al
		denominator1 = 0;
		words_in_this_doc = collect(values(data[time]))
		wordArr = zeros(V)
		
		for word = 1:V
			indices = findin(words_in_this_doc, word)
			tmp=length(indices)
			wordArr[word] = tmp

			numerator2_tmp = soft_lambda[cid][word] + tmp
			numerator2 += lgamma(numerator2_tmp)

			tmp_denominator2 += soft_lambda[cid][word]
			denominator1 += lgamma(soft_lambda[cid][word])
		end

		numerator1 = lgamma(tmp_denominator2)
		denominator2 = lgamma(tmp_denominator2 + length(data[time]))
		posterior = (numerator1+numerator2) - (denominator1+denominator2)

		for k=1:cid-1
			posterior += log(soft_v[k]) - log(soft_v[k] + soft_u[k])
		end
		posterior += log(soft_u[cid]) - log(soft_u[cid]+soft_v[cid])
	end

	return posterior
end

function get_normalized_probabilities(z_posterior_array_probability)
	normalizing_constant = logsumexp(z_posterior_array_probability)
	EXP_z_posterior_array_probability = deepcopy(z_posterior_array_probability)
	EXP_z_posterior_array_probability -= normalizing_constant
	EXP_z_posterior_array_probability = exp(EXP_z_posterior_array_probability)
	return EXP_z_posterior_array_probability
end





function update_statistics(current_support, z_posterior_array_probability, z_posterior_array_cid,  posterior,cid, data, time, soft_lambda, soft_u, soft_v)
	posterior = 1
	wordArr = getWordArr(data, time)
	EXP_z_posterior_array_probability = get_normalized_probabilities(z_posterior_array_probability)
	eta = hyperparameters["eta"]; alpha=hyperparameters["a"]

	max_cid = max(current_support)
	
	for j in current_support 
		is_new_cid = false
		if j == max_cid
			is_new_cid = true
		end

		if j == cid
			###### LAMBDA #######
			for word = 1:V
				if is_new_cid == false #existing cluster
					soft_lambda[cid][word] = soft_lambda[cid][word] + LRATE*(-soft_lambda[cid][word] + eta + NUM_DOCS*(posterior*wordArr[word]))
				else
					soft_lambda[cid][word] =  LRATE*(eta + NUM_DOCS*(posterior*wordArr[word]))
				end			
			end

			###### U ########
			if is_new_cid == false
				soft_u[cid] = soft_u[cid] + LRATE*(-soft_u[cid] + 1 + NUM_DOCS*posterior)
			else
				soft_u[cid] = LRATE*(1 + NUM_DOCS*posterior)
			end	

			###### V #######
			sufficient_stats = 0
			for i=1:length(z_posterior_array_cid)
				if z_posterior_array_cid[i] > cid
					sufficient_stats += EXP_z_posterior_array_probability[i]
				end
			end

			if is_new_cid == false
				soft_v[cid] = soft_v[cid] + LRATE*(-soft_v[cid] + alpha + NUM_DOCS*sufficient_stats)
			else
				soft_v[cid] =  LRATE*(alpha + NUM_DOCS*sufficient_stats)
			end

		else
			if is_new_cid == true
				for word=1:V
					soft_lambda[cid][word] = hyperparameters["eta"]
				end
				soft_u[cid] = 1
				soft_v[cid] = hyperparameters["a"]
			else
				for word=1:V
					soft_lambda[cid][word] = soft_lambda[cid][word] + LRATE*(-soft_lambda[cid][word] + hyperparameters["eta"])
				end
				soft_u[j] = soft_u[j] + LRATE*(-soft_u[j] + 1)
				soft_v[j] = soft_v[j] + LRATE*(-soft_v[j] + hyperparameters["a"])
			end
		end
	end
	return soft_lambda, soft_u, soft_v
end



function get_chibbs(soft_lambda, soft_u, soft_v)
	mean_lambda = deepcopy(soft_lambda); mean_u = deepcopy(soft_u);
	for topic in collect(keys(soft_lambda))
		lambda_Z = sum(collect(values(mean_lambda[topic])))
		for word = 1:V
			mean_lambda[topic][word] /= lambda_Z
		end

		mean_u[topic] /= (mean_u[topic] + soft_v[topic])
	end

	return mean_lambda, mean_u
end


function chibbs_loglikelihood(mean_lambda, mean_u, data, _start, _end)
	logL = 0
	for t = _start:_end
		alltopics = collect(keys(mean_lambda))
		logL_arr_i = zeros(length(alltopics))
		for topic in alltopics
			logL_arr_i[topic] += log(mean_u[topic])
			wordArr = getWordArr(data, t)
			for word=1:V
				logL_arr_i[topic]+= wordArr[word]*log(mean_lambda[topic][word])
			end
		end
		logL += logsumexp(logL_arr_i)
	end
	return logL
end


function get_margin_loglikelihood(prev_weight, prev_support, time, LOOKAHEAD_DELTA, prev_cid, N, data, soft_lambda,soft_u,soft_v)

	VARIATIONAL_ITERATIONS = 5
	for iter=1:VARIATIONAL_ITERATIONS
		c_aggregate = []
		for t=time:time+LOOKAHEAD_DELTA

			current_support = myappend(prev_support, max(prev_support)+1)
			z_posterior_array_probability = []
			z_posterior_array_cid = []
			max_support_value = max(current_support)

			for j in current_support
				zj_probability = sample_q_variational(j, current_support, t, N, max_support_value, soft_lambda, soft_u,soft_v, data)
				z_posterior_array_probability = myappend(z_posterior_array_probability, zj_probability)
				z_posterior_array_cid = myappend(z_posterior_array_cid, j)
			end

			## Choose support (j) by sampling cid from gibbs using mult
			posterior, sampled_cid = sample_cid(z_posterior_array_probability, z_posterior_array_cid)
			soft_lambda, soft_u, soft_v = update_statistics(current_support, z_posterior_array_probability, z_posterior_array_cid, posterior,sampled_cid, data, time, soft_lambda, soft_u, soft_v)
			c_aggregate = myappend(c_aggregate, sampled_cid)

			max_cid = max(current_support)
			if sampled_cid == max_cid
				current_support = myappend(current_support, max_cid )
			end
		end
	end

	mean_lambda, mean_u = get_chibbs(soft_lambda, soft_u, soft_v)
	return chibbs_loglikelihood(mean_lambda, mean_u, data, time, time+LOOKAHEAD_DELTA)
end


end
