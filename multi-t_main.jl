using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("./util/arg.jl")
include("./util/data_pipeline.jl")

using DelimitedFiles
using Random

m_seed = 161803

Random.seed!(m_seed)

model_type, total_eps, n_sim = get_t_main_arg(get_Args())

seeds = rand(000000:999999, Int(ceil(n_sim/Threads.nthreads())*Threads.nthreads()))
# seeds = [001993 109603 619089 071198 383163 213556 410290 908818 123123 456456 789789 012012]
# seeds = [001993 109603 619089]

@info "Using $(Threads.nthreads()) threads for $(length(seeds)) seeds"
@info "Seeds: $seeds"

rewards = Vector{AbstractVector}(undef, Threads.nthreads())

Threads.@threads for i in 1:Threads.nthreads()
  rewards[i] = Vector{AbstractVector}(undef, 0)
end

for (j, total_ep) in enumerate(total_eps)
	@info "Running each experiments for $total_ep episodes"
	Threads.@threads for (i, seed) in collect(enumerate(seeds))
		@info "Starting experiment with seed $i"
		reward = run_exp(seed, model_type; total_eps=total_ep)
		push!(rewards[Threads.threadid()], reward)
		@info "Completed experiment with seed $i of $total_ep episodes"
	end
	io = open("./results/Q$model_type-total_ep=$total_ep.txt", "w") do io
		writedlm(io, hcat(vcat(rewards...)...)')
		@info "Logged all seeded experiments for $total_ep episodes!"
	end
	@info "Completed $(j/length(total_eps)*100)% of steps experiments"
end
