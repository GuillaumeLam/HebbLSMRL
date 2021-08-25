module LSM

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
include("waspnet_ext.jl")



export LSMParams, LSM_Wrapper

end  # module LSM
