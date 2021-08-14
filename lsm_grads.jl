include("./src/lsm.jl")

using WaspNet
using Flux
using Random
using Zygote

seed = 123

function train(network::LSM.LSM_Wrapper, x, y, opt)
    println("Training")

    println("Initial Prediction")
    # ŷ = network(x)
    # println(ŷ)
    # println(y)
    # println("loss")
    # println(loss(ŷ,y))
    println("weight matrix")
    println(network.W_readout)

	θ = Flux.params(network)

    # grads = gradient(() -> loss(ŷ,y), θ)
	# grads = Zygote.gradient(() -> loss(y, network(x)), θ)
	# grads = Zygote.gradient(model -> loss(y, model(x)), network)
	grads = gradient(() -> loss(network(X), Y), θ)

    # w_grads = reshape(collect(keys(grads.params.params.dict)), size(last(network.layers).W))
    # println(w_grads)
    # update!(opt, last(network.layers).W, -reshape(collect(keys(grads.params.params.dict)), size(last(network.layers).W)))

	Flux.Optimise.update!(opt, θ, grads)
	# Flux.Optimise.update!(opt, θ[1, grads[1])

    println("New Better Prediction")
    # ŷ = network(x)
    # println(ŷ)
    # println(y)
    # println("loss")
    # println(loss(ŷ,y))
	loss(y, network(x))
    println("weight matrix")
	println(network.W_readout)

	return grads
end

# loss(ŷ,y) = sum((ŷ .- y).^2)./length(y)

function loss(ŷ, y)
	l = sum((ŷ .- y).^2)./length(y)
	println("loss:")
	println(l)
	return l
end

opt = Descent(0.1)

rng = Random.seed!(seed)
X, Y = rand(rng, 4), rand(rng, 2)
cartpole_param = LSM.LSMParams(8,2,"cartpole")
W = rand(cartpole_param.n_out,cartpole_param.res_out)
lsm = LSM.LSM_Wrapper(cartpole_param, W, rng)

gs = gradient(() -> Flux.mse(lsm(X), Y), params(lsm))
gs[W]
#=
gradient associated with W is currently in the wrong entry, simply need to fix
this issue. Perahps adding an intermediate function in
=#

gs.grads

using ChainRules


ChainRules.rrule(net::WaspNet.AbstractNetwork, x::AbstractVector) =
	net(x), Δ -> (WaspNet.∂lsm∂W(net)*sum(Δ)/length(Δ), nothing)

# Zygote.@adjoint (net::AbstractNetwork)(x::AbstractVector) =
#     (net::AbstractNetwork)(x), Δ -> (∂lsm∂W(net)*sum(Δ)/length(Δ), nothing)

y, back = Zygote._pullback(lsm, X)

back(1)


g = train(lsm, X, Y, opt)

g.params
g.grads


Ŷ = rand(2)

Flux.params(lsm)
g
Flux.Optimise.update!(opt, Flux.params(lsm)[1], g[1])

g = Zygote.gradient(model -> loss(Y, model(X)), lsm)
loss(Y, lsm(X))




using ReinforcementLearning
using ReinforcementLearning: BasicDQNLearner, NamedTuple
import ReinforcementLearning.RLBase.update!

function update!(learner::BasicDQNLearner, batch::NamedTuple{SARTS})

    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_func

    s, a, r, t, s′ = send_to_device(device(Q), batch)
    a = CartesianIndex.(a, 1:length(a))

	println("smth")

    # gs = gradient(Q) do model
    #     q = model(s)[a]
	# 	println()
    #     q′ = vec(maximum(model(s′); dims = 1))
    #     G = r .+ γ .* (1 .- t) .* q′
    #     loss = loss_func(G, q)
    #     Zygote.ignore() do
    #         learner.loss = loss
    #     end
    #     loss
    # end

	gs = gradient(params(Q)) do
        q = Q(s)[a]
        q′ = vec(maximum(Q(s′); dims = 1))
        G = r .+ γ .* (1 .- t) .* q′
        loss = loss_func(G, q)
        Zygote.ignore() do
            learner.loss = loss
        end
        loss
		println(loss)
    end

    # update!(Q, gs)
	Flux.Optimise.update!(Q.optimizer, Flux.params(q)[1], gs[1])
end

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, params(app), gs)


y_t, back_t = Zygote._pullback(loss, lsm(X), Y)

back_t(y_t)

y, back = Zygote._pullback(lsm, X)

back(y)
