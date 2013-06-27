require("random_sum.jl")
arg=Dict()
arg["ta"]=10
arg["va"]=1
nsteps = [arg, arg]

tic()
out = pmap(random_sum, nsteps)

#for i in nsteps
#	random_sum(i)
#end
toc()
println(out[])

