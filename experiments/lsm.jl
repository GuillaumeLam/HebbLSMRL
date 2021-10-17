using WaspNet
using Flux
using Flux.Optimise: update!
using Zygote
using Zygote: @adjoint
using LinearAlgebra
using Random
using SparseArrays
using Distributions

using ReinforcementLearning
using Flux
using Flux.Losses
using StableRNGs

seed = 123
Random.seed!(seed)

function create_conn(val, avg_conn, ub, n)
    if val < avg_conn/n
        return rand()*ub
    else
        return 0.
    end
end

function genPositiveArr!(arr::AbstractVector)
	s = map(arr) do e
		if e < 0
			return [0, abs(e)]
		else
			return [e, 0]
		end
	end

	return vcat(s...)
end

function (net::AbstractNetwork)(x::AbstractVector,sim_τ=0.001, sim_T=0.1)
    poissonST = getPoissonST(genPositiveArr!(x), Bernoulli)
    sim = simulate!(net, poissonST, sim_τ, sim_T)

    return vec(normalize(sum(sim.outputs[end-(length(last(net.prev_outputs))-1):end,:],dims=2), sim_T/sim_τ))
end

using SliceMap

function (net::AbstractNetwork)(m::AbstractMatrix)
    # return mapslices(net, m, dims=1)
    return slicemap(net, m, dims=1)
end

@adjoint (net::AbstractNetwork)(x::AbstractVector) = (net::AbstractNetwork)(x), Δ -> (Δ,Δ)


function train(network::AbstractNetwork, x, y, opt)
    println("Training")

    println("Initial Prediction")
    ŷ = network(x)
    println(ŷ)
    println(y)
    println("loss")
    println(loss(ŷ,y))
    println("weight matrix")
    println(last(network.layers).W)

	θ = Flux.params(network)

    grads = gradient(() -> loss(ŷ,y), θ)
    # w_grads = reshape(collect(keys(grads.params.params.dict)), size(last(network.layers).W))
    # println(w_grads)
    # update!(opt, last(network.layers).W, -reshape(collect(keys(grads.params.params.dict)), size(last(network.layers).W)))

	update!(opt, θ, grads)

    println("New Better Prediction")
    ŷ = network(x)
    println(ŷ)
    println(y)
    println("loss")
    println(loss(ŷ,y))
    println("weight matrix")
    println(last(network.layers).W)
end


struct LSM_Params{I<:Real, F<:Real}
    n_in::I
    res_in::I

    ne::I
    ni::I

    res_out::I
    n_out::I

    K::I
    C::I

    PE_UB::F
    EE_UB::F
    EI_UB::F
    IE_UB::F
    II_UB::F

    LSM_Params(
        n_in::I,res_in::I,ne::I,ni::I,res_out::I,n_out::I,K::I,C::I,
        PE_UB::F,EE_UB::F,EI_UB::F,IE_UB::F,II_UB::F
        ) where {I<:Real,F<:Real} = new{I,F}(n_in,res_in,ne,ni,res_out,n_out,K,C,PE_UB,EE_UB,EI_UB,IE_UB,II_UB)

    LSM_Params(
            n_in::I,res_in::I,ne::I,ni::I,res_out::I,n_out::I,K::I,C::I
            ) where {I<:Number} = LSM_Params(n_in,res_in,ne,ni,res_out,n_out,K,C,0.6,0.005,0.25,0.3,0.01)

    LSM_Params(n_in::I, n_out::I, env::String) where {I<:Real} = (
        if env == "cartpole"
            return LSM_Params(n_in,10,120,30,10,n_out,5,1)
        end
    )
end


function res(params::LSM_Params, seed::Number)
    in_n = [WaspNet.LIF() for _ in 1:params.res_in]
    in_w = randn(MersenneTwister(seed), params.res_in, params.n_in)
    in_w = sparse(in_w)
    in_l = Layer(in_n, in_w)

    res_n = Vector{AbstractNeuron}([WaspNet.LIF() for _ in 1:params.ne])
    append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF()) for _ in 1:params.ni])

    ### res layer weights
    W_in = cat(create_conn.(rand(params.res_in, params.ne),params.K,params.PE_UB,params.res_in), zeros(params.res_in,params.ni), dims=2)'
    W_in = sparse(W_in)

    W_EI = create_conn.(rand(params.ne,params.ni), params.C, params.EI_UB, params.ne)
    W_IE = create_conn.(rand(params.ni,params.ne), params.C, params.IE_UB, params.ne)
    W_EE = W_EI*W_IE
    W_EE[diagind(W_EE)] .= 0.
    W_II = W_IE*W_EI

    W_res = cat(cat(W_EE, W_EI, dims=2),cat(W_IE, W_II, dims=2), dims=1)
    W_res = sparse(W_res)

    res_w = [W_in, W_res]
    conns = [1, 2]

    res = Layer(res_n, res_w, conns)

    ### liquid-out layer weights
    lout_n = [WaspNet.ReLU() for _ in 1:params.res_out]
    W_lout = cat(rand(params.ne, params.res_out), zeros(params.ni, params.res_out), dims=1)'
    W_lout = sparse(W_lout)

    lout_l = Layer(lout_n, W_lout)


    out_n = [WaspNet.ReLU() for _ in 1:params.n_out]
    w = rand(params.n_out, params.res_out)
    out_l = Layer(out_n, w)

    net = Network([in_l, res, lout_l, out_l], params.n_in)

    return net
end

loss(ŷ,y) = sum((ŷ .- y).^2)

opt = Descent(0.1)


x, y = rand(4), rand(2)

cartpole_param = LSM_Params(8,2,"cartpole")

lsm = res(cartpole_param,seed)

train(lsm, x, y, opt)


import CUDA.device
import Flux.params

device(x::AbstractNetwork) = Val(:cpu)

Flux.trainable(model::AbstractNetwork) = (last(model.layers).W,)


rng = StableRNG(seed)
env = CartPoleEnv(; T = Float32, rng = rng)
ns, na = length(state(env)), length(action_space(env))

env_params = LSM_Params(ns*2,na,"cartpole")
cartpole_lsm = res(env_params, seed)

policy = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = cartpole_lsm |> cpu,
                optimizer = opt,
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = loss,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)
stop_condition = StopAfterStep(1000, is_show_progress=!haskey(ENV, "CI"))
hook = TotalRewardPerEpisode()

run(policy, env, stop_condition, hook)
