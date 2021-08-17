using Pkg
Pkg.activate(".")

include("./data_pipeline_util.jl")

using DelimitedFiles

seeds = [001993 109603 619089 071198 383163 213556 410290 908818 123123 456456]

# reward = run_exp(seeds[1])

total_eps = 500

@info "Running Experiments"
for (i, seed) in enumerate(seeds)
	reward = run_exp(seed; total_eps=total_eps)	#; total_eps=500
	@info "Completed $(i/10)% of experiments"
	io = open("./results/QLSM-total_ep=$total_eps.txt", "a") do io
		writedlm(io, reward')
		@info "Logging run"
	end
end
