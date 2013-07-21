#Variational Lookahead 
#Ardavan Saeedi & Tejas Kulkarni

using Debug

using PyCall

@pyimport pylab
@pyimport sklearn.metrics as metrics

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
			if k < max_support_value
				posterior += log(soft_v[k]) - log(soft_v[k] + soft_u[k])
			end
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


function update_helper(original_time, cid, sampled_cid, soft_lambda, soft_u, soft_v, posterior, wordArr, ITER, DELTA_TIME)
	eta = hyperparameters["eta"]; alpha = hyperparameters["a"]
	
	is_new_cid = (has(soft_lambda,cid) == false)

	LEARNING_RATE = 1/ITER
	
	if is_new_cid == true
		soft_lambda[cid] = Dict()
		soft_u[cid]=0; soft_v[cid]=0;
		sufficient_stats = 0  #[[FIXME]]#[[FIXME]]#[[FIXME]]#[[FIXME]]#[[FIXME]]
	else
		sufficient_stats = int(sampled_cid > cid)     #[[FIXME]]#[[FIXME]]#[[FIXME]]#[[FIXME]]#[[FIXME]]
	end

	for word=1:V
		if is_new_cid == true
			soft_lambda[cid][word] = LEARNING_RATE*(eta + (original_time+DELTA_TIME)*(posterior*wordArr[word]))
		else
			soft_lambda[cid][word]=soft_lambda[cid][word] + LEARNING_RATE*(-soft_lambda[cid][word] + eta + (original_time+DELTA_TIME)*(posterior*wordArr[word]))
		end
	end
	soft_u[cid] = soft_u[cid] + LEARNING_RATE*(-soft_u[cid] + 1 + (original_time+DELTA_TIME)*posterior)
	soft_v[cid] = soft_v[cid] + LEARNING_RATE*(-soft_v[cid] + alpha + (original_time+DELTA_TIME)*sufficient_stats)

	return soft_lambda, soft_u, soft_v
end


function update_statistics(original_time, current_support,  posterior,sampled_cid, data, time, soft_lambda, soft_u, soft_v, ITER, CONDITIONAL, DELTA_TIME)
	wordArr = getWordArr(data, time)
	max_cid = max(current_support)

	for cid in current_support
		posterior = -1; flag=-1;

		if CONDITIONAL == 1
			flag = 1; posterior = 1;
		else
			if cid == sampled_cid# && cid == max_cid
				flag = 1; posterior = 1
			end 
			if cid != sampled_cid && cid < max_cid
				flag = 1; 
				posterior = 0
			end
		end

		if flag == 1
			update_helper(original_time, cid, sampled_cid, soft_lambda, soft_u, soft_v, posterior, wordArr, ITER, DELTA_TIME)
		end
	end

	return soft_lambda, soft_u, soft_v
end



function get_chibbs(soft_lambda, soft_u, soft_v)
	mean_lambda = deepcopy(soft_lambda); mean_u = deepcopy(soft_u);
	mean_Z = 0
	for topic in collect(keys(soft_lambda))
		lambda_Z = sum(collect(values(mean_lambda[topic])))
		for word = 1:V
			mean_lambda[topic][word] /= lambda_Z
		end 
		mean_u[topic] /= (mean_u[topic] + soft_v[topic])
		mean_Z += mean_u[topic]
	end

	for topic in collect(keys(mean_u))
		mean_u[topic] /= mean_Z
	end

	return mean_lambda, mean_u
end


function chibbs_loglikelihood(mean_lambda, mean_u, data, _start, _end)
	logL = 0
	for t = _start:_end#_start:_end
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
	#println("CHIBBS: ", logL)

	return logL
end




function initializeStatistics(history_support)
	soft_lambda = Dict(); soft_u = Dict(); soft_v = Dict();
	eta = hyperparameters["eta"]; a = hyperparameters["a"]

	for topic in history_support
		soft_lambda[topic] = Dict()
		for word=1:V
			soft_lambda[topic][word] = eta
		end
		soft_u[topic] = 1
		soft_v[topic] = a
	end
	return soft_lambda, soft_u, soft_v
end




function get_margin_loglikelihood(gibbs_wt, history_c_aggregate, prev_support, time, DELTA_TIME, prev_cid, N, data)
	if DELTA_TIME == 0
		return 0
	end

	history_support = unique(history_c_aggregate)
	soft_lambda, soft_u, soft_v = initializeStatistics(history_support)

	#current_support = myappend(prev_support, max(prev_support)+1)
	current_support = deepcopy(history_support)

	VARIATIONAL_ITERATIONS = 10
	DEBUG = false

	if DEBUG
		println("STATISTICS FROM OUTSIDE LOOKAHEAD")
		println(soft_lambda)
		println(soft_u)
		println(soft_v)
	end

	FIRST_TIME_IN_LOOKAHEAD = 1

	c_aggregate = []
	

	for iter=1:VARIATIONAL_ITERATIONS
		c_aggregate = []
		#println("-=-=-=-=")
		for t=1:time+DELTA_TIME
			############ INSIDE LOOKAHEAD ##########

			if t >= time
				if FIRST_TIME_IN_LOOKAHEAD == 1 && t == time
					FIRST_TIME_IN_LOOKAHEAD = 0
					current_support = myappend(current_support, max(current_support)+1)
					#current_support = [1]
				end
				z_posterior_array_probability = []
				z_posterior_array_cid = []
				max_support_value = max(current_support)
				for j in current_support
					zj_probability = sample_q_variational(j, current_support, t, N, max_support_value, soft_lambda, soft_u,soft_v, data)
					z_posterior_array_probability = myappend(z_posterior_array_probability, zj_probability)
					z_posterior_array_cid = myappend(z_posterior_array_cid, j)
				end

				#if time == 4 && N == 2
				#	@bp
				#end

				## Choose support (j) by sampling cid from gibbs using mult
				posterior, sampled_cid = sample_cid(z_posterior_array_probability, z_posterior_array_cid)
				CONDITIONAL = 0
				"""if t < time
					sampled_cid = history_c_aggregate[t]
					CONDITIONAL = 1
				end"""
			else
			############### OUTSIDE LOOKAHEAD ##############
				sampled_cid = history_c_aggregate[t]
				posterior = 1
				current_support = unique(history_c_aggregate[1:t])#history_support
				CONDITIONAL = 1
			end

			soft_lambda, soft_u, soft_v = 
				update_statistics( time, current_support, posterior,sampled_cid, data, t, 
									soft_lambda, soft_u, soft_v, iter, CONDITIONAL, DELTA_TIME )

			c_aggregate = myappend(c_aggregate, sampled_cid)

			if DEBUG
				println("iter:", iter, " | t=", t, " | word=", data[t][1] , " | current_topic[t]:",sampled_cid ) #" true_topic[t]:", true_topics[t], 
				println("q(z_t):", get_normalized_probabilities(z_posterior_array_probability))
				println(soft_lambda)
				println("\n")
			end

			############ INSIDE LOOKAHEAD ##########
			if t >= time
				if sampled_cid == max_support_value
					current_support = myappend(current_support, max_support_value + 1 )
				end
			end

		end

		#if DEBUG
		#println("--------------------------------------------")
		#println(c_aggregate)
		#println("ITER:",iter, " ARI:", metrics.v_measure_score(c_aggregate, true_topics[1:time+DELTA_TIME]))
		#end
	end

	mean_lambda, mean_u = get_chibbs(soft_lambda, soft_u, soft_v)
	logL = chibbs_loglikelihood(mean_lambda, mean_u, data, time, time+DELTA_TIME)
	
	ARI = metrics.v_measure_score(c_aggregate, true_topics[1:time+DELTA_TIME])

	#println("TRUET:", true_topics[1:time+DELTA_TIME])
	#println("LCAGG:", c_aggregate, " V:", ARI, " LOGL:", logL, " W:", gibbs_wt)

	"""
	misses = 0
	for jj = 1:length(true_topics)
		if c_aggregate[jj] != true_topics[jj]
			misses += 1
		end
	end
	println("MISSES:", misses)"""

	"""ARI = metrics.adjusted_rand_score(c_aggregate, true_topics[1:time+DELTA_TIME])
	println("ARI:", ARI, " CHIBBS:", logL) 
	println("soft_u:", soft_u)
	println("soft_v:", soft_v)
	println("mean_lambda:", mean_lambda)
	println("mean_u:", mean_u)
	println("soft_lambda:", soft_lambda)
	print(history_c_aggregate); print(c_aggregate, "\n\n")"""
	
	return logL
end


end
