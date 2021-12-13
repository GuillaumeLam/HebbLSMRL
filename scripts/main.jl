using Pkg
Pkg.activate(".")
Pkg.instantiate()

using HebbLSMRL

model_type, total_eps, n_sim, parallel = get_main_arg(get_Args())
n_sim = 32
parallel = true

if parallel
	using Distributed
	using SharedArrays
	addprocs(8)
else
	using StableRNGs
end

if parallel
	@everywhere begin
		using Pkg
		Pkg.activate(".")
		Pkg.instantiate()

		using HebbLSMRL

		using DelimitedFiles
		using Random
		using StableRNGs

		using ProgressMeter
	end
end

m_seed = 161803
# m_seed = 314156

m_rng = StableRNG(m_seed)

seeds = rand(m_rng, 000000:999999, n_sim)
rngs = StableRNG.(seeds)

function main(rngs, model_type, total_eps, parallel)
	for (j, total_ep) in enumerate(total_eps)
		@info "Running each experiments for $total_ep episodes"
		isdir("./results") || mkdir("./results")

		if !parallel
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
		else
			@info "Launching parallel exp"

			rewards = pmap((rng)->(HebbLSMRL.run_exp(rng, model_type=model_type, total_eps=total_ep)), rngs)
			# store col first
			io = open("./results/Q$model_type-e=$total_ep.txt", "w") do io
				writedlm(io, hcat(rewards...))
				@info "Logged all seeded experiments for $total_ep episodes!"
			end
		end

		@info "Completed $(j/length(total_eps)*100)% of steps experiments"
	end
end

main(rngs, model_type, total_eps, parallel)
