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
            return LSMParams(n_in,32,120,30,32,n_out,3,4)
        end
    )
end

mutable struct LSM_Wrapper{M<:AbstractMatrix, N<:AbstractNetwork}
    W_readout::M
    reservoir::N

    function (lsm::LSM_Wrapper)(x)
        h = Zygote.ignore() do
            h = lsm.reservoir(x)
            return h
        end
        z = lsm.W_readout*h
        return z
    end

    LSM_Wrapper(W::M, res::N) where {M<:AbstractMatrix, N<:AbstractNetwork} =
        new{M,N}(W, res)

    # function LSM_Wrapper(params::P, rng) where {P<:LSMParams}
    #     reservoir = LSM.init_res(params, rng)
    #     W_readout = rand(rng, params.n_out, params.res_out)
    #     return LSM_Wrapper(W_readout, reservoir)
    # end

    function LSM_Wrapper(params::P, rng::R) where {P<:LSMParams,R<:AbstractRNG}
        reservoir = init_res(params, rng)
        W_readout = rand(rng, params.n_out, params.res_out)
        return LSM_Wrapper(W_readout, reservoir)
    end

    function LSM_Wrapper(params::P, W::M, rng::R) where {P<:LSMParams,M<:AbstractMatrix,R<:AbstractRNG}
        reservoir = init_res(params, rng)
        copy!(W, rand(rng, params.n_out, params.res_out))
        return LSM_Wrapper(W, reservoir)
    end
end


###
# Overloaded functions
###

function (net::AbstractNetwork)(x::AbstractVector,sim_τ=0.001, sim_T=0.1)
    poissonST = WaspNet.getPoissonST(genPositiveArr!(x), Distributions.Bernoulli)
    sim = simulate!(net, poissonST, sim_τ, sim_T)

    idx = length(last(net.prev_outputs))-1

    output_smmd = sum(sim.outputs[end-idx:end,:],dims=2)

    # normalization to be transfered to output layer of LSM_wrapper
    output_nrmlzd = vec(LinearAlgebra.normalize(output_smmd, sim_T/sim_τ))

    return output_nrmlzd
end

function (net::AbstractNetwork)(m::AbstractMatrix)
    # return mapslices(net, m, dims=1)
    return SliceMap.slicemap(net, m, dims=1)
end

# ∂relu(n::Number) = n >= 0 ? 1. : 0.
# ∂relu(l::AbstractVector) = ∂relu.(l)
#
# function ∂lsm∂W(net::AbstractNetwork)
#     h = net.layers[end-1].output
#     return ∂relu(last(net.layers).W * h)*h'
#     # return _∂lsm∂W(last(net.layers).W, h)
# end
#
# # _∂lsm∂W(w,h) = ∂relu(w * h)*h'
#
# Zygote.@adjoint (net::AbstractNetwork)(x::AbstractVector) =
#     (net::AbstractNetwork)(x), Δ -> (∂lsm∂W(net)*sum(Δ)/length(Δ), nothing)


# Flux.trainable(model::AbstractNetwork) = (last(model.layers).W,)
Flux.trainable(lsm::LSM_Wrapper) = (lsm.W_readout,)

# Flux.@functor LSM (W_readout,)

# CUDA.device(x::AbstractNetwork) = Val(:cpu)
CUDA.device(lsm::LSM_Wrapper) = Val(:cpu)


###
# Network Constructor
###

function init_res(params::LSMParams, rng::AbstractRNG)
    ### liquid-in layer creation
    in_n = [WaspNet.LIF() for _ in 1:params.res_in]
    in_w = randn(rng, params.res_in, params.n_in)
    in_w = sparse(in_w)
    in_l = Layer(in_n, in_w)


    ### res layer creation
    res_n = Vector{AbstractNeuron}([WaspNet.LIF() for _ in 1:params.ne])
    append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF()) for _ in 1:params.ni])

    W_in = cat(create_conn.(rand(rng, params.res_in, params.ne),params.K,params.PE_UB,params.res_in), zeros(params.res_in,params.ni), dims=2)'
    W_in = SparseArrays.sparse(W_in)

    W_EI = create_conn.(rand(rng, params.ne,params.ni), params.C, params.EI_UB, params.ne)
    W_IE = create_conn.(rand(rng, params.ni,params.ne), params.C, params.IE_UB, params.ne)
    W_EE = W_EI*W_IE
    W_EE[diagind(W_EE)] .= 0.
    W_II = W_IE*W_EI

    W_res = cat(cat(W_EE, W_EI, dims=2),cat(W_IE, W_II, dims=2), dims=1)
    W_res = SparseArrays.sparse(W_res)

    res_w = [W_in, W_res]
    conns = [1, 2]

    res = Layer(res_n, res_w, conns)

    ### liquid-out layer creation
    lout_n = [WaspNet.ReLU() for _ in 1:params.res_out]
    W_lout = cat(rand(rng, params.ne, params.res_out), zeros(params.ni, params.res_out), dims=1)'
    W_lout = SparseArrays.sparse(W_lout)

    lout_l = Layer(lout_n, W_lout)

    ###
    # out_n = [WaspNet.ReLU() for _ in 1:params.n_out]
    # copy!(w,rand(rng, params.n_out, params.res_out))
    # out_l = Layer(out_n, Matrix(I, params.res_out, params.res_out))

    net = Network([in_l, res, lout_l], params.n_in)

    return net
end
