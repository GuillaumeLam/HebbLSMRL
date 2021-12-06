using Distributed
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

seeds = rand(m_rng, 000000:999999, n_sim)

@info "Seeds: $seeds"

rngs = StableRNG.(seeds)

for (j, total_ep) in enumerate(total_eps)
	@info "Running each experiments for $total_ep episodes"

	rewards = pmap((rng)->(run_exp(rng, model_type=model_type, total_eps=total_ep)), rngs)

	io = open("./results/Q$model_type-total_ep=$total_ep.txt", "w") do io
		writedlm(io, hcat(rewards...)')
		@info "Logged all seeded experiments for $total_ep episodes!"
	end

	@info "Completed $(j/length(total_eps)*100)% of steps experiments"
end
