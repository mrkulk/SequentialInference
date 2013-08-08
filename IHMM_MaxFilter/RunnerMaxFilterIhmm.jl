using Distributions
using Debug
using PyCall

require("maxFilterihmm.jl")
@debug begin
NUM_SEEDS = 100
maxFilter_error_vect = zeros(NUM_SEEDS)
smc_error_vect = zeros(NUM_SEEDS)

for seed = 1:NUM_SEEDS
	
	errors_dict = main(seed)
	println(errors_dict)
	maxFilter_error_vect[seed] = errors_dict["max_filter_error"]
	smc_error_vect[seed] = errors_dict["SMC_error"] 
	println("seed ", seed)
	println("smc error ", errors_dict["SMC_error"] )
	println("maxfilter error ", errors_dict["max_filter_error"])
	println()
end

mean_diff = -sum(maxFilter_error_vect - smc_error_vect)/NUM_SEEDS
println(mean_diff)
end

