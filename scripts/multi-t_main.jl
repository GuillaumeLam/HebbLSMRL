# to be updated

using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("./util/arg.jl")
include("./util/data_pipeline.jl")

using DelimitedFiles
using Random
using StableRNGs


m_seed = 161803
# m_seed = 314156

m_rng = StableRNG(m_seed)

model_type, total_eps, n_sim = get_t_main_arg(get_Args())

seeds = rand(m_rng, 000000:999999, Int(ceil(n_sim/Threads.nthreads())*Threads.nthreads()))

@info "Using $(Threads.nthreads()) threads for $(length(seeds)) seeds"
@info "Seeds: $seeds"

rngs = StableRNG.(seeds)

rewards = Vector{AbstractVector}(undef, Threads.nthreads())

Threads.@threads for i in 1:Threads.nthreads()
  rewards[i] = Vector{AbstractVector}(undef, 0)
end

for (j, total_ep) in enumerate(total_eps)
	@info "Running each experiments for $total_ep episodes"
	Threads.@threads for (i, rng) in collect(enumerate(rngs))
		@info "Starting experiment with seed $(seeds[i])"
		reward = run_exp(rng, model_type; total_eps=total_ep)
		push!(rewards[Threads.threadid()], reward)
		@info "Completed experiment with thread $i of $total_ep episodes"
	end

	io = open("./results/Q$model_type-total_ep=$total_ep.txt", "w") do io
		writedlm(io, hcat(vcat(rewards...)...)')
		@info "Logged all seeded experiments for $total_ep episodes!"
	end

	Threads.@threads for i in 1:Threads.nthreads()
	  rewards[i] = Vector{AbstractVector}(undef, 0)
	end

	@info "Completed $(j/length(total_eps)*100)% of steps experiments"
end
