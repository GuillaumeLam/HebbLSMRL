using BenchmarkTools
using StableRNGs

include("./data_pipeline.jl")

lsm = model_dict["LSM"](4,2,StableRNG(123))

x = rand(4)

# @btime lsm(x)   # ~5.329 ms (208617 allocations: 9.80 MiB)

in = lsm.preprocessing(x)
# @btime lsm.reservoir(in)    # ~5.008 ms (208601 allocations: 9.80 MiB)
                            # reservoir is bottle neck

using Distributions
using LinearAlgebra

# poissonST = WaspNet.getPoissonST(in, Distributions.Bernoulli) # ~411.814 ns (7 allocations: 416 bytes)
#
# # ~5.306 ms (208573 allocations: 9.71 MiB) for 0.1s sim
# # ~57.116 ms (1972574 allocations: 94.30 MiB) for 1s sim
# sim = simulate!(lsm.reservoir, poissonST, 0.001, 0.1)
#
# # ~18.894 μs (20 allocations: 97.47 KiB)
# @btime begin
#     idx = length(last(lsm.reservoir.prev_outputs))-1
#
#     output_smmd = sum(sim.outputs[end-idx:end-Int(0.2*(idx+1)),:],dims=2)
#
#     if all(output_smmd.==0)
#         return vec(output_smmd)
#     else
#         return vec(LinearAlgebra.normalize(output_smmd, 0.1/0.001))
#     end
# end


using .LSM
using Random
using SparseArrays

using WaspNet

Random.seed!(123)


params = LSMParams(8,2,"cartpole")
rng = StableRNG(123)

# function init_res(params::LSMParams, rng::AbstractRNG)


lif_params = Float32.((20., 10., 0.5, 0., 0., 0., 0.))

### liquid-in layer creation
in_n = [WaspNet.LIF(lif_params...) for _ in 1:params.res_in]
in_w = Float32.(randn(rng, params.res_in, params.n_in))
in_w = sparse(in_w)
in_l = Layer(in_n, in_w, type=Float32)


### res layer creation
res_n = Vector{AbstractNeuron}([WaspNet.LIF(lif_params...) for _ in 1:params.ne])
append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF(lif_params...)) for _ in 1:params.ni])

W_in = cat(LSM.create_conn.(rand(rng, params.res_in, params.ne),params.K,params.PE_UB,params.res_in), zeros(params.res_in,params.ni), dims=2)'
W_in = SparseArrays.sparse(Float32.(W_in))

W_EI = LSM.create_conn.(rand(rng, params.ne,params.ni), params.C, params.EI_UB, params.ne)
W_IE = LSM.create_conn.(rand(rng, params.ni,params.ne), params.C, params.IE_UB, params.ne)
W_EE = W_EI*W_IE
W_EE[diagind(W_EE)] .= 0.
W_II = W_IE*W_EI

W_res = cat(cat(W_EE, W_EI, dims=2),cat(W_IE, W_II, dims=2), dims=1)
W_res = SparseArrays.sparse(Float32.(W_res))

res_w = [W_in, W_res]
conns = [1, 2]

res = Layer(res_n, res_w, conns, type=Float32)

net = Network([in_l, res], params.n_in, type=Float32)

#     return net
# end

@btime net(in)
