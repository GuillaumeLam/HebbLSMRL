# to be updated

using Revise
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using StableRNGs
seeds = [809669]

include("./misc/experiments.jl")

# using RL_LSM

model_type, total_eps = "RL_LSM", 50

frames = nothing

for (j, total_ep) in enumerate(total_eps)
	for (i, seed) in enumerate(seeds)
		reward, frames = run_exp(StableRNG(seed), model_type; total_eps=total_ep)
	end
end


using Plots

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
cart(x) = rectangle(0.4,0.1,x-0.2,0)

function pole(x,θ)
	free_pt = (x+sin(θ)*0.5, cos(θ)*0.5)
	connected_pt = (x,0.1)
	return Shape([free_pt, connected_pt, connected_pt, free_pt])
end

function cartpole_frame(x,θ)
	plot(cart(x), xlim=(-3,3), ylim=(0,1), color=:black, legend=false)
	plot!(pole(x, θ), xlim=(-3,3), ylim=(0,1), legend=false)
end


anim = @animate for row in eachrow(frames["env"])
	cartpole_frame(row[1], row[3])
end

gif(anim, "cartpole_env-QLSM.gif", fps=20)
