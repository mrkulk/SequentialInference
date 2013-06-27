#Tejas Kulkarni - tejask@mit.edu
# DPMM with SMC
#Julia PATH: 
#	/Users/tejas/Documents/julia/julia

using Distributions
using Debug
using PyCall
@pyimport pylab
@pyimport sklearn.metrics as metrics

@debug begin

############# HELPER FUNCTIONS and DATASTRUCTURES #################
myappend{T}(v::Vector{T}, x::T) = [v..., x] #Appending to arrays
NUM_PARTICLES = 500
DIMENSIONS = 1
NUM_POINTS = 99
state = Dict()
particles = Dict()
hyperparameters = Dict()
hyperparameters["a"]=1;hyperparameters["b"]=1;hyperparameters["alpha"]=0.5;hyperparameters["tao"]=5*5;hyperparameters["eta"]=0;
DEBUG = 0
data = Dict()


#################### DATA LOADER AND PLOTTING ##################################
COLORS =[[rand(),rand(),rand()] for i =1:50]

function plotPoints(data,fname)
	for i=1:length(data)
		pylab.plot(data[i][1],data[i][2], "o", color=COLORS[data[i]["c"]])
	end
	pylab.savefig(string(fname,".png"))
end

function plotPointsfromChain(time)
	ariArr = []
	for N=1:length(particles[time])
		"""for i=1:time
			pylab.plot(data[i][1],data[i][2], "o", color=COLORS[particles[time][N]["hidden_state"]["c_aggregate"][i]])
		end
		pylab.savefig(string("time:", time, " PARTICLE_",N,"_",".png"))"""

		true_clusters = data["c_aggregate"][1:time]
		inferred_clusters = particles[time][N]["hidden_state"]["c_aggregate"]
		ariArr = myappend(ariArr, metrics.adjusted_rand_score(inferred_clusters, true_clusters))
	end
	println("time:", time," Maximum ARI: ", max(ariArr))
end

function loadObservations()
	data = Dict()
	mu={[0,0], [2,2], [4,4]}
	std={[0.25,0.25], [0.25,0.25], [0.25,0.25]}
	data["c_aggregate"] = zeros(NUM_POINTS)

	data["get_data_arr"] = Dict()
	for d=1:DIMENSIONS
		data["get_data_arr"][d]=[]
	end

	for i=1:NUM_POINTS
		sample_arr = rand(Multinomial(1,[1/2,1/6,1/3]))
		idx = findin(sample_arr, 1)[1]
		data[i] = Dict()
		data[i]["c"] = idx
		data["c_aggregate"][i] = idx
		for d=1:DIMENSIONS
			data[i][d] = rand(Normal(mu[idx][d],std[idx][d]))
			data["get_data_arr"][d] = myappend(data["get_data_arr"][d], data[i][d])
		end
	end
	#plotPoints(data,"original")
	return data
end



#################### MAIN FUNCTION DEFINITIONS ####################
function normalizeWeights(time)
	normalizing_constant = sum([s["weight"] for s in values(particles[time])])
	for i = 1:length(particles[time])
		particles[time][i]["weight"]/=normalizing_constant
		#particles[time][i]["weight"] = exp(particles[time][i]["weight"]) #exponentiating to get probability values
	end
end


## devised by Fearnhead and Clifford (2003)
function FC_resample(time)
	weight_vector = [s["weight"] for s in values(particles[time])]
	weight_vector = float64(weight_vector)
	particles_new_indx = int(zeros(length(particles[time])))
	particles_temporary = copy(particles[time])

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
	weight_vector = [s["weight"] for s in values(particles[time])]
	weight_vector = float64(weight_vector)
	particles_new_indx = int(zeros(length(particles[time])))
	particles_temporary = copy(particles[time])
	
	particles[time] = Dict() ##?? 

	for i = 1:NUM_PARTICLES
		sample_arr = rand(Multinomial(1,weight_vector))
		particles_new_indx[i] = findin(sample_arr, 1)[1]
		particles[time][i] = particles_temporary[particles_new_indx[i]]
		particles[time][i]["weight"] = 1/NUM_PARTICLES
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


function posterior_z_helper(nj, total_pts, a, b, tao, alpha ,eta, obs_mean, obs_var)
	posterior = 0
	posterior += a*log(b) + log(gamma(a+nj*0.5))
	posterior -= log(gamma(a)) + log(sqrt(nj*tao + 1))
	
	tmp_term = 0
	t1 = (obs_mean-eta); t1=t1.*t1
	tmp_term += log( b + (nj*(obs_var+t1/(1+nj*tao))*0.5) )
	posterior += -(a+nj*0.5)*tmp_term
	return posterior
end



function posterior_z_j_old(support, state,time)
	a = hyperparameters["a"]; b=hyperparameters["b"]; alpha = hyperparameters["alpha"]; tao = hyperparameters["tao"]; eta = hyperparameters["eta"];total_pts = time
	posterior = 0

	for cid in support
		cid_cardinality, indices = get_pts_in_cluster(state["c_aggregate"], cid)
		posterior += log(cid_cardinality/(total_pts + alpha)) ##prior
		#println("[posterior_z_j_old]", " v:", posterior, " cid:", cid_cardinality)
		for d=1:DIMENSIONS
			obs_mean = get_empirical_mean(data["get_data_arr"][d][indices])#[1:time])
			obs_var = get_empirical_variance(data["get_data_arr"][d][indices], obs_mean)#[1:time], obs_mean)
			posterior += posterior_z_helper(cid_cardinality, total_pts, a, b, tao, alpha ,eta, obs_mean, obs_var)
		end
	end

	return posterior
end



function posterior_z_j_new(support, state,time)
	a = hyperparameters["a"]; b=hyperparameters["b"]; alpha = hyperparameters["alpha"]; tao = hyperparameters["tao"]; eta = hyperparameters["eta"];total_pts = time
	posterior = 0

	for cid in support 
		if cid < max(state["c_aggregate"])
			cid_cardinality, indices = get_pts_in_cluster(state["c_aggregate"], cid)
			posterior += log(cid_cardinality/(total_pts + alpha)) ##prior
			#println("[posterior_z_j_new] existing", " v:", posterior, " cid:", cid_cardinality)
		else #new cluster
			cid_cardinality = 1
			indices = time
			posterior += log(alpha/(total_pts + alpha)) ##prior
			#println("[posterior_z_j_new] new", " v:", posterior, " cid:", cid_cardinality)
		end

		for d=1:DIMENSIONS
			obs_mean = get_empirical_mean(data["get_data_arr"][d][indices])#[1:time])
			obs_var = get_empirical_variance(data["get_data_arr"][d][indices], obs_mean)#[1:time], obs_mean)
			posterior += posterior_z_helper(cid_cardinality, total_pts, a, b, tao, alpha ,eta, obs_mean, obs_var)
		end
	end

	return posterior
end

## deleting ancestors as do not need them now
function recycle(time)
	if time >= 3
		delete!(particles,time-2)
	end
end


function run_sampler()
	#### particle init ####
	state=Dict()
	state["c"] = 1
	state["c_aggregate"] = [1]
	time = 1
	particles[time] = Dict() #time = 0
	for i = 1:NUM_PARTICLES
		particles[time][i] = Dict() #partile_id = 0
		particles[time][i] = {"weight" => 1, "hidden_state" => state}
	end
	normalizeWeights(time)
	FC_resample(time)
	for time = 2:NUM_POINTS

		println("##################")
		println("time: ", time)
		###### PARTICLE CREATION and EVOLUTION #######
		particles[time]=Dict()

		PARTICLE_COUNT = 1
		for N=1:length(particles[time-1])

			###  Calculate posterior of older state
			support = particles[time-1][N]["hidden_state"]["c_aggregate"]
			posterior_old = posterior_z_j_old(support, particles[time-1][N]["hidden_state"], time-1)
			
			if DEBUG == 1
				println("PARTICLE:", N ," weight:", particles[time-1][N]["weight"], " support:",support)		
			end

			### Creating particles with different support
			z_support = particles[time-1][N]["hidden_state"]["c_aggregate"]
			z_support = myappend(z_support, max(z_support)+1)
			for j in z_support
				state=Dict()
				state["c"] = j
				state["c_aggregate"] = myappend(particles[time-1][N]["hidden_state"]["c_aggregate"], j)
				particles[time][PARTICLE_COUNT] = Dict(); 
				particles[time][PARTICLE_COUNT]["hidden_state"] = state;

				new_support = particles[time][PARTICLE_COUNT]["hidden_state"]["c_aggregate"];

				ratio = posterior_z_j_new(new_support, particles[time][PARTICLE_COUNT]["hidden_state"], time) - posterior_old;
				if particles[time-1][N]["weight"] == 0.0
					@bp
				end		
				weight =  log(particles[time-1][N]["weight"]) - ratio
				weight = exp(weight)
				particles[time][PARTICLE_COUNT]["weight"] = weight+1e-50; 
				PARTICLE_COUNT+=1 #this will be eventually equal to N*max(support of all prev particles)
			end
		end
		normalizeWeights(time)
		FC_resample(time)
		recycle(time)
		if mod(time, 1) == 0
			plotPointsfromChain(time)
		end
	end

end


#################### MAIN RUNNER ####################
loadObservations()
run_sampler()



end #debug




