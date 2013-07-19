#Tejas Kulkarni - tejask@mit.edu
# DPMM with SMC
#Julia PATH: 
#	/Applications/Julia.app/Contents/Resources/julia/bin/julia

using Distributions
using Debug
#using PyCall
#@pyimport pylab

@debug begin

############# HELPER FUNCTIONS and DATASTRUCTURES #################
myappend{T}(v::Vector{T}, x::T) = [v..., x] #Appending to arrays
NUM_PARTICLES = 2
DIMENSIONS = 2
NUM_POINTS = 100
state = Dict()
particles = Dict()
hyperparameters = Dict()
hyperparameters["a"]=1;hyperparameters["b"]=1;hyperparameters["alpha"]=0.5;hyperparameters["tao"]=5*5;hyperparameters["eta"]=0;

#################### DATA LOADER AND PLOTTING ##################################

# function plotPoints(data)
# 	colors = {"red","blue","grey"}
# 	for i=1:length(data)
# 		pylab.plot(data[i][1], data[i][2], color=colors[data[i]["c"]])
# 	pylab.savefig("foo.png")
# end

function loadObservations()
	data = Dict()
	mu={[0,0], [2,2], [4,4]}
	std={[0.25,0.25], [0.25,0.25], [0.25,0.25]}

	for i=1:NUM_POINTS
		sample_arr = rand(Multinomial(1,[1/2,1/6,1/3]))
		idx = findin(sample_arr, 1)[1]
		data[i] = Dict()
		data[i]["c"] = idx
		for d=1:DIMENSIONS
			data[i][d] = rand(Normal(mu[idx][d],std[idx][d]))
		end
	end
	#plotPoints(data)
	return data
end



#################### MAIN FUNCTION DEFINITIONS ####################
function normalizeWeights(time)
	normalizing_constant = sum([s["weight"] for s in values(particles[time])])
	for i = 1:length(particles[time])
		particles[time][i]["weight"]/=normalizing_constant
	end
end


function resample(time)
	weight_vector = [s["weight"] for s in values(particles[time])]
	weight_vector = float64(weight_vector)
	particles_new_indx = int(zeros(length(particles[time])))
	particles_temporary = copy(particles[time])

	for i = 1:NUM_PARTICLES
		sample_arr = rand(Multinomial(1,weight_vector))
		particles_new_indx[i] = findin(sample_arr, 1)[1]
		particles[time][i] = copy(particles_temporary[particles_new_indx[i]])
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
	posterior -= log(gamma(a)) + log(sqrt(nj*tao) + 1)
	
	tmp_term = 0
	t1 = (obs_mean-eta); t1=t1.*t1
	tmp_term += log( b + (nj*(obs_var+t1/(1+nj*tao))*0.5) )
	posterior += -(a+nj*0.5)*tmp_term
	#if posterior == 0
	#	@bp
	#end
	return posterior
end




function posterior_z_j(cid, _type, state,time)
	println("#########")
	a = hyperparameters["a"]; b=hyperparameters["b"]; alpha = hyperparameters["alpha"]; tao = hyperparameters["tao"]; eta = hyperparameters["eta"];total_pts = time
	posterior = 0
	if cid <= max(state["c_aggregate"])# || _type == "old" #existing cluster
		cid_cardinality, indices = get_pts_in_cluster(state["c_aggregate"], cid)
		posterior += log(cid_cardinality/(total_pts + alpha)) ##prior
		println("existing", " v:", posterior, " cid:", cid_cardinality, " type:", _type )
	else #new cluster
		cid_cardinality = 1
		posterior += log(alpha/(total_pts + alpha)) ##prior
		println("existing", " v:", posterior, " cid:", cid_cardinality, " type:", _type )
	end

	for d=1:DIMENSIONS
		obs_mean = get_empirical_mean(OBSERVATIONS[time][d])
		obs_var = get_empirical_variance(OBSERVATIONS[time][d], obs_mean)
		posterior += posterior_z_helper(cid_cardinality, total_pts, a, b, tao, alpha ,eta, obs_mean, obs_var)
	end
	println("BEFORE", " v:", posterior)
	posterior += 1e-100
	println("AFTER", " v:", posterior)
	return posterior
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
	resample(time)

	for time = 2:length(OBSERVATIONS)
		###### PARTICLE CREATION and EVOLUTION #######
		particles[time]=Dict()

		max_cid = -1
		tmparr = [s[2]["hidden_state"]["c_aggregate"] for s in particles[1]]
		newarr = []
		for elm in tmparr
			elm_max = max(elm)
			if max_cid <= elm_max
				max_cid = elm_max
			end
		end

		for j = 1:max_cid + 1
			state=Dict()
			state["c"] = j
			state["c_aggregate"] = myappend(particles[time-1][j]["hidden_state"]["c_aggregate"], j)
			particles[time][j] = Dict()
			particles[time][j]["hidden_state"] = state

			new_support = particles[time][j]["hidden_state"]["c_aggregate"]
			old_support = particles[time-1][j]["hidden_state"]["c_aggregate"]

			ratio = posterior_z_j(j, "new", particles[time][j]["hidden_state"], time) - posterior_z_j(j, "old", particles[time-1][j]["hidden_state"], time-1)
			println(ratio, " ", exp(ratio), " maxcid:",j)
			weight =  particles[time-1][j]["weight"] * exp(ratio)
			particles[time][j]["weight"] = weight
		end
		normalizeWeights(time)
		resample(time)
	end

end


#################### MAIN RUNNER ####################
OBSERVATIONS = loadObservations()
run_sampler()



end #debug




