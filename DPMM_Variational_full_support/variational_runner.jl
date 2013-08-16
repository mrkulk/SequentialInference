#Tejas Kulkarni & Ardavan Saeedi
#tejasdkulkarni@gmail.com | tejask@mit.edu
#DPMM with SMC
#Julia PATH: 
#	/Users/tejas/Documents/julia/julia


using Distributions
using Debug
using PyCall
using Base.Collections
using NumericExtensions
require("dataset.jl")
require("variational_lookahead.jl")
require("gradient.jl")
require("maxfilter.jl")

using NumericExtensions

@pyimport pylab
@pyimport sklearn.metrics as metrics


type lookaheadOBJ
	zj_probability
	current_c_aggregate
	time
	current_support
	N
end

@debug begin 

############# HELPER FUNCTIONS and DATASTRUCTURES #################
myappend{T}(v::Vector{T}, x::T) = [v..., x] #Appending to arrays


const ENUMERATION = 0

#const NUM_PARTICLES = 1
#LOOKAHEAD_DELTA = 10#10
#const INTEGRAL_PATHS = 2#2

#Interesting seeds: 10(delta=20), 109
#srand(133) #133-b #10 #109

WORDS_PER_DOC = 200
NUM_DOCS = 100	#200
NUM_TOPICS = NaN
V = NaN
state = Dict()
particles = Dict()
hyperparameters = Dict()
hyperparameters["eta"]=0.5;hyperparameters["a"]=1;hyperparameters["lrate"] = 1
const _DEBUG = 0
data = Dict()
true_topics = []


LRATE = hyperparameters["lrate"]

#################### DATA LOADER AND PLOTTING ##################################
"""
function plotPoints(data,fname)
	for i=1:NUM_POINTS
		pylab.plot(data[i][1],data[i][2], "o", color=COLORS[data[i]["c"]])
	end
	pylab.savefig(string(fname,".png"))
end"""

function plotPointsfromChain(time)
	ariArr = []
	pylab.clf()
	for N=1:length(particles[time])
		"""for i=1:time
			pylab.plot(data[i][1],data[i][2], "o", color=COLORS[particles[time][N]["hidden_state"]["c_aggregate"][i]])
		end
		pylab.savefig(string("time:", time, " PARTICLE_",N,"_",".png"))"""

		true_clusters = data["c_aggregate"][1:time]
		inferred_clusters = particles[time][N]["hidden_state"]["c_aggregate"]
		ariArr = myappend(ariArr, metrics.v_measure_score(inferred_clusters, true_clusters))
		
		#println("------")
		#println("TRUE:", true_clusters)
		#println("INFR:", inferred_clusters)
		#println("------")
	end
	if length(ARGS) == 0
	#	println("time:", time," Maximum ARI: ", max(ariArr))
	end
	return max(ariArr)
end



function loadObservations()
	data = Dict()
	theta, pi, V = dataset1()

	#NUM_DOCS = length(topics)
	true_topics = []
	
	syn_topics=[]
	"""syn_topics=int(zeros(NUM_DOCS))
	syn_topics[1:2:2000]=1;
	syn_topics[1001:1500]=2;
	syn_topics[1501:2000]=3;"""


	data["c_aggregate"] = int(zeros(NUM_DOCS))
	
	for i = 1:NUM_DOCS
		data[i] = Dict() #Create doc
		if length(syn_topics) == 0
			topic = rand(Multinomial(1,pi)); topic = findin(topic, 1)[1]
		else
			topic = syn_topics[i]
		end

		#topic = rand(Multinomial(1,pi)); topic = findin(topic, 1)[1]

		true_topics=myappend(true_topics, topic)
		data["c_aggregate"][i] = topic
		for j = 1:WORDS_PER_DOC
			data[i][j] = rand(Multinomial(1, theta[topic])); data[i][j] = findin(data[i][j], 1)[1] 
		end
	end

	return data
end



#################### MAIN FUNCTION DEFINITIONS ####################
function normalizeWeights(time)
	norm_arr = zeros(length(particles[time]))
	for i = 1:length(particles[time])
		norm_arr[i] = particles[time][i]["weight"]
	end
	normalizing_constant = logsumexp(norm_arr)

	for i = 1:length(particles[time])
		particles[time][i]["weight"]-=normalizing_constant
		particles[time][i]["weight"] = exp(particles[time][i]["weight"])
	end
end


## devised by Fearnhead and Clifford (2003)
function FC_resample(time)
	weight_vector = [s["weight"] for s in values(particles[time])]
	weight_vector = float64(weight_vector)
	particles_new_indx = int(zeros(length(particles[time])))
	particles_temporary = deepcopy(particles[time])

	particles[time] = Dict() ##?? 

	for i = 1:NUM_PARTICLES
		if particles_temporary[i]["weight"] < 1/50
			sample_arr = rand(Multinomial(1,weight_vector))
			particles_new_indx[i] = findin(sample_arr, 1)[1]
			particles[time][i] = particles_temporary[particles_new_indx[i]]
			particles[time][i]["weight"] = 1/50
		else
			particles[time][i] = particles_temporary[i]
		end
	end
end	



function resample(time)
	weight_vector=zeros(NUM_PARTICLES)
	for i=1:NUM_PARTICLES
		weight_vector[i] = particles[time][i]["weight"]
	end
	particles_new_indx = int(zeros(length(particles[time])))
	particles_temporary = deepcopy(particles[time])
	
	particles[time] = Dict() ##?? 

	for i = 1:NUM_PARTICLES
		sample_arr = rand(Multinomial(1,weight_vector))
		particles_new_indx[i] = findin(sample_arr, 1)[1]
		particles[time][i] = particles_temporary[particles_new_indx[i]]
		#particles[time][i]["weight"] = 1/NUM_PARTICLES
	end
end	




function get_empirical_mean(y)
	return sum(y)/length(y)
end

function get_empirical_variance(y, empirical_mean)
	diff = (y - empirical_mean)
	diff = diff .* diff
	return sum(diff)/length(y)
end

function get_pts_in_cluster(clusters,cid) #number of cid's in clusters
	indices = findin(clusters,cid)
	return length(indices), indices
end



function get_joint_crp_probability(cid, cid_cardinality, indices, alpha)
	numerator = 0
	denominator = 0
	for i=1:cid_cardinality-1
		numerator += log(i)
		denominator += log(alpha + i - 1)
	end
	denominator += log(alpha + cid_cardinality - 1)

	ret = log(alpha)+numerator-denominator

	return ret
end




## deleting ancestors as do not need them now
function recycle(time)
	if time >= 3
		delete!(particles,time-2)
	end
end



function sample_cid(z_posterior_array_probability, z_posterior_array_cid)
	normalizing_constant = logsumexp(z_posterior_array_probability)

	EXP_z_posterior_array_probability = deepcopy(z_posterior_array_probability)
	EXP_z_posterior_array_probability -= normalizing_constant
	EXP_z_posterior_array_probability = exp(EXP_z_posterior_array_probability)

	sample_arr = rand(Multinomial(1,EXP_z_posterior_array_probability))
	indx = findin(sample_arr, 1)[1]
	cid = z_posterior_array_cid[indx]
	weight = z_posterior_array_probability[indx]

	return weight, cid
end


function initWordArr(data)
	for t=1:NUM_DOCS
		words_in_this_doc = collect(values(data[t]))
		wordArr = zeros(V)
		for word = 1:V
			indices = findin(words_in_this_doc, word)
			tmp=length(indices)
			wordArr[word] = tmp
		end
		GLOBAL_WORD_ARR[t] = wordArr
	end
end

function getWordArr(data, time)
	return GLOBAL_WORD_ARR[time]
	"""
	words_in_this_doc = collect(values(data[time]))
	wordArr = zeros(V)
	for word = 1:V
		indices = findin(words_in_this_doc, word)
		tmp=length(indices)
		wordArr[word] = tmp
	end
	return wordArr"""
end


function existing_topic_posterior_helper(time, N, eta, cid, prior)

	state = particles[time-1][N]["hidden_state"]
	
	numerator1 = 0; tmp_denominator1 = 0; #this is first side page 5 from Chong et al
	numerator2 = 0; tmp_denominator2 = 0; #this is second side page 5 from Chong et al
	denominator1 = 0;

	words_in_this_doc = collect(values(data[time]))
	wordArr = zeros(V)

	lambda_sufficient_stats=zeros(V)

	for word = 1:V
		indices = findin(words_in_this_doc, word)
		
		tmp=length(indices)
		wordArr[word] = tmp

		numerator2_tmp = state["lambda"][cid][word] + tmp
		numerator2 += lgamma(numerator2_tmp)

		tmp_denominator2 += state["lambda"][cid][word]
		denominator1 += lgamma(state["lambda"][cid][word])

		lambda_sufficient_stats[word] = numerator2_tmp
	end

	numerator1 = lgamma(tmp_denominator2)
	denominator2 = lgamma(tmp_denominator2 + length(data[time]))

	posterior = prior + (numerator1+numerator2) - (denominator1+denominator2)

	return posterior, lambda_sufficient_stats
end


function get_posterior_zj(cid, c_aggregate,time, N, root_support) 

	eta = hyperparameters["eta"]; alpha=hyperparameters["a"]; total_pts = time
	posterior = 0
	lambda_sufficient_stats = NaN

	new_cluster_flag = 0
	if cid < max(root_support)
		cid_cardinality, indices = get_pts_in_cluster(c_aggregate, cid)
		posterior += log(cid_cardinality/(total_pts + alpha)) ##prior
		#println("[PRIOR] existing", " value:", exp(posterior), " cid:", cid, "cid_cardinality:", cid_cardinality)
	else #new cluster
		cid_cardinality = 1
		new_cluster_flag = 1
		posterior += log(alpha/(total_pts + alpha)) ##prior
		#println("[PRIOR] new", " value:", exp(posterior), " cid:", cid, "cid_cardinality:", cid_cardinality)
	end

	if new_cluster_flag == 1 #new cluster
		numerator1 = lgamma(eta*V)
		denominator1 = V*lgamma(eta)
		posterior += numerator1 - denominator1 
		numerator2 = 0;
		wordArr = zeros(V)
		for word = 1:V
			words_in_this_doc = collect(values(data[time]))
			indices = findin(words_in_this_doc, word)
			tmp=length(indices)
			wordArr[word] = tmp
			numerator2 += lgamma(eta + tmp)
		end
		denominator2 = lgamma(eta*V + length(data[time]))
		
		posterior += numerator2		
		posterior -= denominator2

		#println("[[[[NEW]]]]:", numerator1,"  ||  ",  denominator1,"  ||  ",  numerator2,"  ||  ", denominator2)

	else #existing cluster
		posterior, lambda_sufficient_stats = existing_topic_posterior_helper(time, N,eta,cid, posterior)
	end

	#println("[POSTERIOR] ", posterior , " v:", exp(posterior), " cid:", cid)
	#println("\n")
	return posterior, lambda_sufficient_stats

end




function path_integral(time, N, do_lookahead)

	root_support = particles[time-1][N]["hidden_state"]["c_aggregate"]
	root_support = unique(root_support)

	max_root_support= max(root_support)

	if particles[time-1][N]["hidden_state"]["c_aggregate"][time-1] == max_root_support
		root_support = unique(myappend(root_support, max_root_support+1))
	end

	max_root_support=max(root_support)
	wordArr = getWordArr(data,time)
	z_posterior_array_probability = []; z_posterior_array_cid = []; lambda_sufficient_stats_ARR = Dict();
	support_array = [];

	cnt = 1
	for j in root_support
		current_c_aggregate = myappend(particles[time-1][N]["hidden_state"]["c_aggregate"], j)
		zj_probability, lambda_sufficient_stats = get_posterior_zj(j, current_c_aggregate, time, N, root_support)

		if time < NUM_DOCS && DELTA > 0 && do_lookahead
			DELTA_TIME = min(LOOKAHEAD_DELTA, NUM_DOCS - time - 1)
			#println(DELTA_TIME, " ", NUM_DOCS - time)
			history = myappend(particles[time-1][N]["hidden_state"]["c_aggregate"], j)
			prev_support = unique(history)
			lookahead_logprobability = get_margin_loglikelihood(zj_probability, history, prev_support, time+1, DELTA_TIME, j, N, data)
			zj_probability += lookahead_logprobability
		end

		z_posterior_array_probability = myappend(z_posterior_array_probability, zj_probability)
		z_posterior_array_cid = myappend(z_posterior_array_cid, j)
		support_array = myappend(support_array, max_root_support)
		#lambda_sufficient_stats_ARR[j] =  lambda_sufficient_stats
		lambda_sufficient_stats_ARR[cnt] =  lambda_sufficient_stats; cnt+=1
	end
	
	"""weight, sampled_cid = sample_cid(z_posterior_array_probability, z_posterior_array_cid)

	if sampled_cid == max_root_support #has(particles[time][N]["hidden_state"]["lambda"], sampled_cid) == true
		update_newcluster_statistics(sampled_cid, data,time,wordArr, weight, N)
	else
		update_existingcluster_statistics(sampled_cid, data,time,wordArr, weight, N, lambda_sufficient_stats_ARR[sampled_cid])
	end

	return weight, sampled_cid"""
	return z_posterior_array_probability, z_posterior_array_cid, lambda_sufficient_stats_ARR, support_array
end



function update_lambda_after_resample(time, lambda_stats_putative_array)
	wordArr = getWordArr(data,time)

	for i = 1:NUM_PARTICLES
		weight = particles[time][i]["weight"]
		sampled_cid = particles[time][i]["hidden_state"]["c"]

		if sampled_cid == particles[time][i]["max_support"]
			update_newcluster_statistics(sampled_cid, data,time,wordArr, weight, i)
		else
			update_existingcluster_statistics(sampled_cid, data,time,wordArr, weight, i, lambda_stats_putative_array[particles[time][i]["hidden_state"]["sampled_indx"]])
		end
	end
end

function putativeResample(time, particles_t, particles_t_minus_1, log_maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, support_array_putative)
	normalizing_constant = logsumexp(log_maxfilter_probability_array)
	normalized_probabilities = log_maxfilter_probability_array
	normalized_probabilities -=  normalizing_constant
	normalized_probabilities = exp(normalized_probabilities)

	#print("time:", time)
	#println(maxfilter_cid_array)
	#println(normalized_probabilities)

	for i = 1:NUM_PARTICLES
		sample_arr = rand(Multinomial(1,normalized_probabilities))
		sampled_indx = findin(sample_arr, 1)[1]

		particles_t[i]["hidden_state"]["c"] = maxfilter_cid_array[sampled_indx]
		particles_t[i]["hidden_state"]["sampled_indx"] = sampled_indx
		particles_t[i]["hidden_state"]["c_aggregate"] = myappend(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["c_aggregate"], particles_t[i]["hidden_state"]["c"])
		particles_t[i]["weight"] = log_maxfilter_probability_array[sampled_indx]
		particles_t[i]["max_support"] = support_array_putative[sampled_indx] #######################################################
		
		particles_t[i]["hidden_state"]["lambda"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["lambda"])
		particles_t[i]["hidden_state"]["soft_lambda"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["soft_lambda"])
		particles_t[i]["hidden_state"]["soft_u"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["soft_u"])
		particles_t[i]["hidden_state"]["soft_v"] = deepcopy(particles_t_minus_1[maxfilter_particle_struct[sampled_indx]]["hidden_state"]["soft_v"])

		#print(state["c"]," | ")
	end
	#println()
end



function run_sampler()
	#### particle init ####
	state=Dict()
	state["c"] = [1]
	state["c_aggregate"] = [1]

	state["lambda"] = Dict(); state["lambda"][1] = Dict();
	state["soft_lambda"] = Dict(); state["soft_u"] = Dict(); state["soft_v"] = Dict();
	state["soft_lambda"][1] = Dict(); state["soft_u"][1] = Dict(); state["soft_v"][1] = Dict();

	for word = 1:V
		state["lambda"][1][word] = hyperparameters["eta"]
	end

	time = 1
	particles[time] = Dict() #time = 0
	for i = 1:NUM_PARTICLES
		particles[time][i] = Dict() #partile_id = 0
		particles[time][i] = {"weight" => 1, "hidden_state" => state}

		#### update lambda for current document ####
		words_in_this_doc = data[1]
		for word = 1:V
			indices = findin(words_in_this_doc, word)
			tmp=length(indices)
			numerator2_tmp = state["lambda"][1][word] + tmp
			particles[time][i]["hidden_state"]["lambda"][1][word] = numerator2_tmp
			particles[time][i]["hidden_state"]["soft_lambda"][1][word] = hyperparameters["eta"] + LRATE*(NUM_DOCS*tmp)
		end
		particles[time][i]["hidden_state"]["soft_u"][1] = 1 + LRATE*NUM_DOCS
		particles[time][i]["hidden_state"]["soft_v"][1] = hyperparameters["a"]
	end
	normalizeWeights(time)
	resample(time)


	for time = 2:NUM_DOCS

		if length(ARGS) == 0
			#println("##################")
			println("time: ", time)
		end

		###### PARTICLE CREATION and EVOLUTION #######
		particles[time]=Dict()
		maxfilter_probability_array = []
		maxfilter_cid_array = []
		maxfilter_particle_struct=[]
		
		log_maxfilter_probability_array = []
		support_array_putative = []
		lambda_stats_putative_array=Dict()
		is_new_cid_array = []
		
		do_lookahead = true

		cnt = 1
		for N=1:NUM_PARTICLES

			if _DEBUG == 1
				println("PARTICLE:", N ," weight:", particles[time-1][N]["weight"], " support:",support)		
			end

			particles[time][N] = Dict()
			particles[time][N]["hidden_state"] = Dict();
			particles[time][N]["hidden_state"]["lambda"] = deepcopy(particles[time-1][N]["hidden_state"]["lambda"])
			particles[time][N]["hidden_state"]["soft_lambda"] = deepcopy(particles[time-1][N]["hidden_state"]["soft_lambda"])
			particles[time][N]["hidden_state"]["soft_u"] = deepcopy(particles[time-1][N]["hidden_state"]["soft_u"])
			particles[time][N]["hidden_state"]["soft_v"] = deepcopy(particles[time-1][N]["hidden_state"]["soft_v"])

			z_posterior_array_probability, z_posterior_array_cid, lambda_sufficient_stats_ARR, support_array = path_integral(time,N, do_lookahead)
			for ii=1:length(z_posterior_array_probability)
				maxfilter_probability_array = myappend(maxfilter_probability_array, exp(z_posterior_array_probability[ii]))
				log_maxfilter_probability_array = myappend(log_maxfilter_probability_array, z_posterior_array_probability[ii])
				maxfilter_cid_array = myappend(maxfilter_cid_array, z_posterior_array_cid[ii])
				maxfilter_particle_struct = myappend(maxfilter_particle_struct, N)
				support_array_putative = myappend(support_array_putative, support_array[ii])
				lambda_stats_putative_array[cnt] = lambda_sufficient_stats_ARR[ii]
				cnt+=1
			end
		end

		if MAXFILTERING == 1
			if EQUIVALENCE_MAXFILTERING == 1
				stratifiedMaxFiltering(time, particles[time], deepcopy(particles[time-1]), maxfilter_probability_array, maxfilter_cid_array,maxfilter_particle_struct, NUM_PARTICLES, log_maxfilter_probability_array, support_array_putative)	
			else
				maxFilter(particles[time], deepcopy(particles[time-1]), maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, NUM_PARTICLES)
			end
		else
			putativeResample(time, particles[time], deepcopy(particles[time-1]), log_maxfilter_probability_array, maxfilter_cid_array, maxfilter_particle_struct, support_array_putative)
			#normalizeWeights(time)
			#resample(time)
			#recycle(time)
		end

		update_lambda_after_resample(time, lambda_stats_putative_array)
		recycle(time)

		#println(particles)
		if mod(time, NUM_DOCS) == 0
			return plotPointsfromChain(time)
		end
	end

end


#################### MAIN RUNNER ####################
LOOKAHEAD_DELTA=0
if length(ARGS) > 0
	NUM_PARTICLES = int(ARGS[1])
	DELTA = int(ARGS[2])
	SEED = int(ARGS[3])
	srand(SEED)
else
	NUM_PARTICLES = 20#1
	DELTA = 15#15#50#20 will return without lookahead
	SEED = 12
	srand(SEED)
end

MAXFILTERING=1
EQUIVALENCE_MAXFILTERING=1

data = loadObservations()

GLOBAL_WORD_ARR=Dict()
initWordArr(data)

#for ttt = 1:10
#print("WITHOUT LOOKAHEAD: ")
#NUM_PARTICLES = 1
LOOKAHEAD_DELTA = 0
ari_without_lookahead = 0#run_sampler()


#print("\nWITH LOOKAHEAD: ")
#NUM_PARTICLES = 1
LOOKAHEAD_DELTA = DELTA
ari_with_lookahead = run_sampler()

print([ari_without_lookahead, ari_with_lookahead])
#end

end

