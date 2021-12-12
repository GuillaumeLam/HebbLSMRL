using Pkg
Pkg.activate("..")
Pkg.instantiate()

using HebbLSMRL

# include("../arg.jl")
# include("./rl-rc/data_pipeline.jl")

using DelimitedFiles
using StableRNGs

# seeds = [001993 109603 619089 071198 383163 213556 410290 908818 123123 456456]
seeds = [809669]
rngs = StableRNG.(seeds)

model_type, total_eps = get_main_arg(get_Args())
total_eps = 100

function main(rngs, model_type, total_eps)
	for (j, total_ep) in enumerate(total_eps)
		@info "Running each experiments for $total_ep episodes"
		isdir("./results") || mkdir("./results")

		frame = Matrix{Float64}(undef, total_ep, length(rngs))

		for (i, rng) in enumerate(rngs)
			@info "Starting experiment $i"
			# reward = run_exp(StableRNG(seed), model_type=model_type, total_eps=total_ep)
			HebbLSMRL.run_exp!(rng, frame[:,i], model_type=model_type, total_eps=total_ep)
			@info "Completed $(i/length(rngs)*100)% of experiments of $total_ep episodes"
			# frame = hcat(frame, reward)
			GC.gc()
		end

		# store col first
		io = open("./results/Q$model_type-e=$total_ep.txt", "a") do io
			writedlm(io, frame)
			@info "Logged runs!"
		end

		@info "Completed $(j/length(total_eps)*100)% of steps experiments"
	end
end

@btime main(rngs, model_type, total_eps)
# => 1 rng
# 14.705 s (292777993 allocations: 13.44 GiB)
# => 2 rngs
# 63.560 s (608922502 allocations: 27.98 GiB)
# => 3 rngs
# 84.030 s (909077645 allocations: 41.74 GiB)
