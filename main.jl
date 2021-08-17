using Pkg
Pkg.activate(".")

include("./data_pipeline_util.jl")

using DelimitedFiles

seeds = [001993 109603 619089 071198 383163 213556 410290 908818 123123 456456]

# reward = run_exp(seeds[1])

#todo: run bottom code for array of total eps ie to generate all data
total_eps = 500

@info "Running Experiments"
for (i, seed) in enumerate(seeds)
	@info "Starting experiment $i"
	reward = run_exp(seed; total_eps=total_eps)	#; total_eps=500
	@info "Completed $(i/length(seeds)*100)% of experiments"
	io = open("./results/QLSM-total_ep=$total_eps.txt", "a") do io
		writedlm(io, reward')
		@info "Logged run!"
	end
end
