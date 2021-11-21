module HebbLSMRL

using ArgParse
using DelimitedFiles
using Flux
using LiquidStateMachine
using Random
using ReinforcementLearning
using StableRNGs
using Statistics

include("arg.jl")

export get_main_arg, get_t_main_arg, get_Args

include("experiments.jl")
include("results_visual.jl")
include("timing.jl")

export run_exp

end # module
