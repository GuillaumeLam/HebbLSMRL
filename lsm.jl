using WaspNet
using Flux
using Zygote
using Zygote: @adjoint
using LinearAlgebra
using Random
using SparseArrays



function create_conn(val, avg_conn, ub, n)
    if val < avg_conn/n
        return rand()*ub
    else
        return 0.
    end
end

seed = 123
Random.seed!(seed)

n_in = 10
K = 5

ne = 120
ni = 30
C = 1.

n_lout = 10
n_out = 4

PE_UB = 0.6
EE_UB = 0.05
EI_UB = 0.25
IE_UB = 0.3
II_UB = 0.01

in_n = [WaspNet.LIF() for _ in 1:n_in]
in_w = randn(MersenneTwister(seed), n_in, n_in)
in_w = sparse(in_w)
in_l = Layer(in_n, in_w)

res_n = Vector{AbstractNeuron}([WaspNet.LIF() for _ in 1:ne])
append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF()) for _ in 1:ni])

### res layer weights
W_in = cat(create_conn.(rand(n_in,ne), K, PE_UB, n_in), zeros(n_in,ni), dims=2)'
W_in = sparse(W_in)

W_EI = create_conn.(rand(ne,ni), C, EI_UB, ne)
W_IE = create_conn.(rand(ni,ne), C, IE_UB, ne)
W_EE = W_EI*W_IE
W_EE[diagind(W_EE)] .= 0.
W_II = W_IE*W_EI

W_res = cat(cat(W_EE, W_EI, dims=2),cat(W_IE, W_II, dims=2), dims=1)
W_res = sparse(W_res)

res_w = [W_in, W_res]
conns = [1, 2]

res = Layer(res_n, res_w, conns)

### liquid-out layer weights
lout_n = [WaspNet.ReLU() for _ in 1:n_lout]
W_lout = cat(rand(ne, n_lout), zeros(ni, n_lout), dims=1)'
W_lout = sparse(W_lout)

lout_l = Layer(lout_n, W_lout)


out_n = [WaspNet.ReLU() for _ in 1:n_out]
w = rand(n_out, n_lout)
out_l = Layer(out_n, w)

lsm = Network([in_l, res, lout_l, out_l], n_in)

@btime simulate!(lsm, poissonST([1,1,1,1,1,1,1,1,1,1]), 0.001, 0.1, track_state=true)
# takes 29.255 ms to simulate 100ms with reservoir for cartpole
# takes 79.518 ms to simulate 250ms with reservoir for cartpole
# takes 335.399 ms to simulate 1s with reservoir for cartpole

# sim = simulate!(lsm, poissonST([1,1,1,1,1,1,1,1,1,1]), 0.001, 0.1, track_state=true)

# argmax(sum(sim.outputs[end-3:end,:], dims=2))

for i in 1:5:40
    display(plot(sim.states[i:i+4,:]', layout = (5, 1)))
end

# function LSM(params)
#     return (NE, NI) -> initLSM(ne, ni, params)
# end
#
# function initLSM(NE::AbstractNeuron, NI::AbstractNeuron)
#
# end

function (net::AbstractNetwork)(x::AbstractVector,sim_τ=0.001, sim_T=0.1)
    sim = simulate!(net, poissonST(x), sim_τ, sim_T)

    return normalize(sum(sim.outputs[end-3:end,:],dims=2), sim_T/sim_τ)
end

# lsm([1,1,1,1,1,1,1,1,1,1])

θ = Params(w)

loss(ŷ,y) = sum((ŷ .- y).^2)

x, y = rand(10), rand(4)
# l = loss(lsm(x),y)

@adjoint lsm(x) = lsm(x), Δ -> (Δ, Δ)

# grads = gradient(() -> loss(lsm(x),y), θ)
#
# typeof(grads)
# grads[w]
# grads.params.params
# reshape(collect(keys(grads.params.params.dict)), (n_out, n_lout))

using Flux.Optimise: update!

# update!(w, 0.1 *reshape(collect(keys(grads.params.params.dict)), (n_out, n_lout)))
#
# collect(keys(grads.params.params.dict))
#
# copy!(grads.)

opt = Descent(0.1)

function train(network::AbstractNetwork, x, y, θ, opt)
    println("Training")
    println("loss")
    println(loss(network(x),y))
    println("weight matrix")
    println(w)
    grads = gradient(() -> loss(network(x),y), θ)
    update!(opt, w, -reshape(collect(keys(grads.params.params.dict)), (n_out, n_lout)))
    println("loss")
    println(loss(network(x),y))
    println("weight matrix")
    println(w)
end

train(lsm, x, y, θ, opt)
