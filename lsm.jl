using WaspNet
using Flux
using Flux.Optimise: update!
using Zygote
using Zygote: @adjoint
using LinearAlgebra
using Random
using SparseArrays

seed = 123
Random.seed!(seed)

function create_conn(val, avg_conn, ub, n)
    if val < avg_conn/n
        return rand()*ub
    else
        return 0.
    end
end

function (net::AbstractNetwork)(x::AbstractVector,sim_τ=0.001, sim_T=0.1)
    sim = simulate!(net, poissonST(x), sim_τ, sim_T)

    return normalize(sum(sim.outputs[end-(length(last(net.prev_outputs))-1):end,:],dims=2), sim_T/sim_τ)
end


@adjoint (net::AbstractNetwork)(x::AbstractVector) = (net::AbstractNetwork)(x), Δ -> (Δ,Δ)


function train(network::AbstractNetwork, x, y, θ, opt)
    println("Training")

    println("Initial Prediction")
    ŷ = network(x)
    println(ŷ)
    println(y)
    println("loss")
    println(loss(ŷ,y))
    println("weight matrix")
    println(last(network.layers).W)

    grads = gradient(() -> loss(ŷ,y), θ)
    w_grads = reshape(collect(keys(grads.params.params.dict)), size(last(network.layers).W))
    println(w_grads)
    update!(opt, last(network.layers).W, -reshape(collect(keys(grads.params.params.dict)), size(last(network.layers).W)))

    println("New Better Prediction")
    ŷ = network(x)
    println(ŷ)
    println(y)
    println("loss")
    println(loss(ŷ,y))
    println("weight matrix")
    println(last(network.layers).W)
end


struct LSMParams{I<:Real, F<:Real}
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

    LSMParams(
        n_in::I,res_in::I,ne::I,ni::I,res_out::I,n_out::I,K::I,C::I,
        PE_UB::F,EE_UB::F,EI_UB::F,IE_UB::F,II_UB::F
        ) where {I<:Real,F<:Real} = new{I,F}(n_in,res_in,ne,ni,res_out,n_out,K,C,PE_UB,EE_UB,EI_UB,IE_UB,II_UB)

    LSMParams(
            n_in::I,res_in::I,ne::I,ni::I,res_out::I,n_out::I,K::I,C::I
            ) where {I<:Number} = LSMParams(n_in,res_in,ne,ni,res_out,n_out,K,C,0.6,0.005,0.25,0.3,0.01)

    LSMParams(n_in::I, n_out::I, env::String) where {I<:Real} = (
        if env == "cartpole"
            return LSMParams(n_in,10,120,30,10,n_out,5,1)
        end
    )
end


function res(params::LSMParams, seed::Number)
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
    θ = Params(w)

    return net, θ
end

loss(ŷ,y) = sum((ŷ .- y).^2)

x, y = rand(4), rand(2)

cartpole_param = LSMParams(4,2,"cartpole")

opt = Descent(0.1)

lsm, θ = res(cartpole_param,seed)

train(lsm, x, y, θ, opt)
