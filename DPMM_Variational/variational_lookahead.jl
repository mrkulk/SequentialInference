#Variational Lookahead 
#Ardavan Saeedi & Tejas Kulkarni


function sample_q_variational(j, current_support, current_c_aggregate, time, N, max_support_value, soft_lambda, soft_u,soft_v, data)
	eta = hyperparameters["eta"]

	if j == max_support_value #new cluster
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

		for k=1:j-1
			posterior += log(soft_v[k]) - log(soft_v[k] + soft_u[k])
		end
		posterior += log(soft_u[j]) - log(soft_u[j]+soft_v[j])
	end

	return posterior
end



function get_margin_loglikelihood(prev_weight, prev_support, prev_c_aggregate, time, prev_cid, N, data,  soft_lambda,soft_u,soft_v)

	#initialize data structures
	q_probabilities=Dict()
	for i=time:time+LOOKAHEAD_DELTA
		soft_lambda[i]=Dict(); soft_u[i]=Dict();soft_v[i]=Dict()
		for w=1:V
			soft_lambda[i][w]=Dict();
		end
		q_probabilities[i]=Dict()
	end

	VARIATIONAL_ITERATIONS = 5
	for iter=1:VARIATIONAL_ITERATIONS
		for t=time:time+LOOKAHEAD_DELTA

			current_support = myappend(prev_support, max(prev_support)+1)
			z_posterior_array_probability = []
			z_posterior_array_cid = []
			max_support_value = max(current_support)

			for j in current_support
				current_c_aggregate = myappend(prev_c_aggregate, j)
				zj_probability = sample_q_variational(j, current_support, current_c_aggregate, t, N, max_support_value, soft_lambda, soft_u,soft_v, data)
				z_posterior_array_probability = myappend(z_posterior_array_probability, zj_probability)
				z_posterior_array_cid = myappend(z_posterior_array_cid, j)
			end

			## Choose support (j) by sampling cid from gibbs using mult
			sampled_probability, sampled_cid = sample_cid(z_posterior_array_probability, z_posterior_array_cid)
			update_statistics(sampled_probability,sampled_cid, data, time, soft_lambda, soft_u, soft_v)
			prev_c_aggregate = myappend(prev_c_aggregate, sampled_cid)
		end
	end

end
