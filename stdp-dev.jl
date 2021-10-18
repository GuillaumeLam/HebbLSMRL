using Revise
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using StableRNGs
seeds = [809669]

include("./misc/experiments.jl")

# using RL_LSM

model_type, total_eps = "RL_LSM", 50

frames = nothing

for (j, total_ep) in enumerate(total_eps)
	for (i, seed) in enumerate(seeds)
		reward, frames = run_exp(StableRNG(seed), model_type; total_eps=total_ep)
	end
end
