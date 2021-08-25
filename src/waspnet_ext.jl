mutable struct LSM_Wrapper{N<:AbstractNetwork}
    readout
    reservoir::N
    preprocessing::Function
    st_gen::SpikeTrainGenerator

    function (lsm::LSM_Wrapper)(x)
        h = Zygote.ignore() do
            x̃ = lsm.preprocessing(x)
            st = lsm.st_gen(x̃)
            return lsm.reservoir(st)
        end
        z = lsm.readout(h)
        return z
    end

    LSM_Wrapper(readout, res::N, func) where {N<:AbstractNetwork} =
        new{N}(readout, res, func, SpikeTrainGenerator(Distributions.Bernoulli))
    LSM_Wrapper(readout, res::N, func, rng) where {N<:AbstractNetwork} =
        new{N}(readout, res, func, SpikeTrainGenerator(Distributions.Bernoulli, rng))
end

function LSM_Wrapper(params::P, rng::R, func) where {P<:LSMParams,R<:AbstractRNG}
    reservoir = init_res(params, rng)
    readout = Chain(Dense(rand(rng,params.res_out, params.ne), rand(rng, params.res_out) ,relu),
        Dense(rand(rng, params.n_out, params.res_out), rand(rng, params.n_out)))
    return LSM_Wrapper(readout, reservoir, func, rng)
end

function LSM_Wrapper(params::P, readout, rng::R, func) where {P<:LSMParams,R<:AbstractRNG}
    reservoir = init_res(params, rng)
    return LSM_Wrapper(readout, reservoir, func, rng)
end

LSM_Wrapper(params::P, rng::R) where {P<:LSMParams,R<:AbstractRNG} = LSM_Wrapper(params, rng, identity)
LSM_Wrapper(params::P, readout, rng::R) where {P<:LSMParams,R<:AbstractRNG} = LSM_Wrapper(params::P, readout, rng, identity)


###
# Overloaded functions
###

function (net::AbstractNetwork)(spike_train_generator, sim_τ=0.001, sim_T=0.1)
    sim = simulate!(net, spike_train_generator, sim_τ, sim_T)

    idx = length(last(net.prev_outputs))-1

    output_smmd = sum(sim.outputs[end-idx:end-Int(0.2*(idx+1)),:],dims=2)

    if all(output_smmd.==0)
        return vec(output_smmd)
    else
        return vec(LinearAlgebra.normalize(output_smmd, sim_T/sim_τ))
    end
end

function (net::AbstractNetwork)(m::AbstractMatrix)
    v = map(net, vec(m))
    return hcat(v...)
end

Flux.trainable(lsm::LSM_Wrapper) = (lsm.readout,)

CUDA.device(lsm::LSM_Wrapper) = Val(:cpu)


###
# Network Constructor
###

function init_res(params::LSMParams, rng::AbstractRNG)
    lif_params = (20., 10., 0.5, 0., 0., 0., 0.)

    ### liquid-in layer creation
    in_n = [WaspNet.LIF(lif_params...) for _ in 1:params.res_in]
    in_w = randn(rng, params.res_in, params.n_in)
    in_w = sparse(in_w)
    in_l = Layer(in_n, in_w)


    ### res layer creation
    res_n = Vector{AbstractNeuron}([WaspNet.LIF(lif_params...) for _ in 1:params.ne])
    append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF(lif_params...)) for _ in 1:params.ni])

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

    net = Network([in_l, res], params.n_in)

    return net
end
