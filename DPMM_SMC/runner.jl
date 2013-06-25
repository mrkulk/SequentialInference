#Tejas Kulkarni - tejask@mit.edu
# DPMM with SMC
#Julia PATH: 
#	/Applications/Julia.app/Contents/Resources/julia/bin/julia

using Distributions
using Debug

#@debug begin

############# HELPER FUNCTIONS and DATASTRUCTURES #################
myappend{T}(v::Vector{T}, x::T) = [v..., x] #Appending to arrays
NUM_PARTICLES = 0 #changes every iteration
DIMENSIONS = 0
NUM_POINTS = 100
state = Dict()
particles = Dict()




#################### DATA LOADER ##################################
function loadObservations()
	data = Dict()
	data[0] = rand(MultivariateNormal(zeros(5), eye(5)))
	return data
end
OBSERVATIONS = loadObservations()




#################### MAIN FUNCTION DEFINITIONS ####################
function normalizeWeights(time)
	normalizing_constant = sum([s["weight"] for s in values(particles[time])])
	for i = 1:NUM_PARTICLES
		particles[time][i]["weight"]/=normalizing_constant
	end
end


function resample(time)
	weight_vector = [s["weight"] for s in values(particles[time])]
	weight_vector = float64(weight_vector)
	particles_new_indx = int(zeros(length(particles[time])))
	particles_temporary = copy(particles[time])

	for i = 1:length(particles[time])
		sample_arr = rand(Multinomial(1,weight_vector))
		particles_new_indx[i] = findin(sample_arr, 1)[1]
	end

	for i =1:length(particles[time])
		if i != particles_new_indx[i]
			particles[time][i] = copy(particles_temporary[particles_new_indx[i]])
		end
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


function posterior_z_helper(cid_cardinality, total_pts, a, b, tao, alpha ,eta)
	posterior += (a*log(b) + log(gamma(a+nj*0.5))
	posterior -= log(gamma(a) + log(sqrt(nj*tao) + 1)
	
	tmp_term = 0
	t1 = (obs_mean-eta); t1=t1.*t1
	tmp_term += log( b + (nj*(obs_var+t1/(1+nj*tao))*0.5) )
	posterior += -(a+nj*0.5)*tmp_term
	return posterior
end




function posterior_z_j(support, state,time)
	a = state["a"]; b=state["b"]; alpha = state["alpha"]; tao = state["tao"]; eta = state["eta"];total_pts = time
	posterior = 1e-100

	for cid in support
		if cid <= max(state["c_aggregate"]) #existing cluster
			cid_cardinality, indices = get_pts_in_cluster(clusters, cid)
			obs_mean = get_empirical_mean(OBSERVATIONS[indices])
			obs_var = get_empirical_variance(OBSERVATIONS[indices], obs_mean)
			posterior += log(cid_cardinality/(total_pts + alpha)) ##prior
			posterior += posterior_z_helper(cid_cardinality, total_pts, a, b, tao, alpha ,eta)
		else #new cluster
			cid_cardinality = 1
			obs_mean = get_empirical_mean(OBSERVATIONS[time])
			obs_var = get_empirical_variance(OBSERVATIONS[time], obs_mean)
			posterior += log(alpha/(total_pts + alpha)) ##prior
			posterior += posterior_z_helper(cid_cardinality, total_pts, a, b, tao, alpha ,eta)
		end
	return exp(posterior)
end



function run_sampler()
	state=Dict()
	state["c"] = 1
	state["c_aggregate"] = [1]
	particles[0] = Dict() #time = 0
	particles[0][0] = Dict() #partile_id = 0
	particles[0][0]["weight"] = 1
	particles[0][0]["hidden_state"] = state
	
	NUM_PARTICLES = 1
	for i = 1:NUM_PARTICLES #for time 0
		particles[0][i] = {"weight" => 1, "hidden_state" => state}
	end

	for time = 1:length(OBSERVATIONS)
		particles[time] = Dict()
		for j = 1:max(particle[time-1]["hidden_state"]["c_aggregate"]) + 1
			state=Dict(); 
			state["c"] = j
			state["c_aggregate"] = myappend(particles[time-1][j]["hidden_state"]["c_aggregate"], j)
			particles[time][j]["hidden_state"] = state

			new_support = state[time]["c_aggregate"]
			old_support = state[time-1]["c_aggregate"]
			weight =  particles[time-1][j]["weight"] * ( posterior_z_j(new_support, state[time], time) / posterior_z_j(old_support, state[time-1], time-1) )
			particles[time][j]["weight"] = weight

		normalizeWeights(time)
		resample(time)
end


#################### MAIN RUNNER ####################
run_sampler()

#end #debug

