module HebbLSMRL

using ArgParse
using DelimitedFiles
using Flux
using LiquidStateMachine
using Random
using ReinforcementLearning
using StableRNGs
using Statistics
using StatsBase
using StatsPlots
using Plots; gr()

include("arg.jl")
export arg, get_Args

include("experiments.jl")
export run_exp

include("results_visual.jl")
export plot_run!, plot_aggregate

end # module
