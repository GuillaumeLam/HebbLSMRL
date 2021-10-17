include("./src/lsm.jl")

using WaspNet
using Flux
using Random
using Zygote

seed = 123

function train(network::RL_LSM.LSM, x, y, opt)
    println("Training")

    println("Initial Prediction")

	θ = Flux.params(network)

	grads = gradient(() -> loss(network(X), Y), θ)

	Flux.Optimise.update!(opt, θ, grads)

    println("New Better Prediction")
	loss(y, network(x))

	return grads
end

function loss(ŷ, y)
	l = sum((ŷ .- y).^2)./length(y)
	println("loss:")
	println(l)
	return l
end

opt = Descent(0.1)

rng = Random.seed!(seed)
X, Y = rand(rng, 4), rand(rng, 2)
cartpole_param = RL_LSM.LSM_Params(8,2,"cartpole")
lsm = RL_LSM.LSM(cartpole_param, rng)

lsm(X)

gs = gradient(() -> Flux.mse(lsm(X), Y), params(lsm))
gs.grads


train(lsm, X, Y, opt)
