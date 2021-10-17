module RL_LSM

using CUDA
using Distributions
using Flux
using LinearAlgebra
using Random
using SliceMap
using SparseArrays
using WaspNet
using Zygote

include("util.jl")
include("spike_generator.jl")
include("lsm-wrapper.jl")



export LSM_Params, LSM

end  # module RL_LSM
