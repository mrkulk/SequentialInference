###Dirichlet Process Mixture Model - Gaussian Observations
==============

This repo contains gibbs and SMC implementation of DPMM. Gibbs sampler is written in python and SMC sampler is written in
Julia. Julia was partly used as a learning exercise but more importantly to explore speed up due to its LLVM-JIT compilation.



#### Gibbs Sampler
DP Mixture Algorithm #2 from Neal 2000 

To run: 
- python DPMM_Gibbs/main.py

#### SMC Sampler
To run:
- Julia DPMM_SMC/runner.jl
