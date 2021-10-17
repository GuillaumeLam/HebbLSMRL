using Pkg
Pkg.activate(".")
Pkg.instantiate()

using StableRNGs
seeds = [809669]

include("./rl-rc/data_pipeline.jl")

model_type, total_eps = "RL_LSM", 100

for (j, total_ep) in enumerate(total_eps)
	# @info "Running each experiments for $total_ep episodes"
	for (i, seed) in enumerate(seeds)
		# @info "Starting experiment $i"
		reward = run_exp(StableRNG(seed), model_type; total_eps=total_ep)
		# @info "Completed $(i/length(seeds)*100)% of experiments of $total_ep episodes"
		# io = open("./results/Q$model_type-total_ep=$total_ep.txt", "a") do io
		# 	writedlm(io, reward')
		# 	@info "Logged run!"
		# end
	end
end
