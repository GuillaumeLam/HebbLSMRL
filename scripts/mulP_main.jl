using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Distributed
using SharedArrays
addprocs(2)

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

m_seed = 161803
# m_seed = 314156

m_rng = StableRNG(m_seed)

model_type, total_eps, n_sim = get_t_main_arg(get_Args())

total_eps = 100
n_sim = 4

seeds = rand(m_rng, 000000:999999, n_sim)
rngs = StableRNG.(seeds)

function main_multP(rngs, model_type, total_eps)
	for (j, total_ep) in enumerate(total_eps)
		@info "Running each experiments for $total_ep episodes"
		isdir("./results") || mkdir("./results")

		frame = SharedArray{Float64}(total_ep, n_sim)

		rewards = pmap((i,rng)->(run_exp!(rng, frame[:,i], model_type=model_type, total_eps=total_ep)), enumerate(rngs))

		# store col first
		io = open("./results/Q$model_type-e=$total_ep.txt", "w") do io
			writedlm(io, hcat(rewards...))
			@info "Logged all seeded experiments for $total_ep episodes!"
		end

		@info "Completed $(j/length(total_eps)*100)% of steps experiments"
	end
end

main_multP(rngs, model_type, total_eps)
#todo
#fix
