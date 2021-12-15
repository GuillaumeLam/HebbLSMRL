module HebbLSMRL

using ArgParse
using DelimitedFiles
using Flux
using LiquidStateMachine
using ReinforcementLearning
using Random
using Statistics
using StatsBase
using StatsPlots
using Plots

include("arg.jl")
export args, get_Args

include("experiments.jl")
export run_exp, run_exp!, exp

include("results_visual.jl")
export plot_run!, plot_aggregate

end # module
