###Dirichlet Process Mixture Model - Gaussian Observations
==============

This repo contains gibbs, SMC and Variational implementation of DPMM. Gibbs sampler is written in python and other samplers are written in
Julia. Julia was partly used as a learning exercise but more importantly to explore speed up due to its LLVM-JIT compilation.



#### Gibbs Sampler
DP Mixture Algorithm #2 from Neal 2000 

To run: 
- python DPMM_Gibbs/main.py

#### SMC Sampler
To run:
- Julia DPMM_SMC/runner.jl

#### Max Filtering (Algorithm proposed by Sam Gershman)
To run:
- Julia DPMM_MaxFilter/runner.jl

#### Variational Particle Filtering for lookahead
To run:
- Julia DPMM_Variational/variational_runner.jl 
