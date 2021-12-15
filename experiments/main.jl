# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()

using HebbLSMRL
using Distributed
using StableRNGs

if parallel
	addprocs(8)

	@everywhere begin
		using Pkg
		Pkg.activate(".")
		Pkg.instantiate()

		using HebbLSMRL
		using StableRNGs
	end
end

model_type, total_eps, n_sim, parallel = args(get_Args())

m_seed = 161803
# m_seed = 314156
m_rng = StableRNG(m_seed)

seeds = rand(m_rng, 000000:999999, n_sim)
rngs = StableRNG.(seeds)

HebbLSMRL.plot_aggregate()
return
HebbLSMRL.exp(rngs, total_eps, parallel, model_type)
