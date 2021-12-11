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
using Plots
# using PyPlot

include("arg.jl")
export get_main_arg, get_t_main_arg, get_Args

include("experiments.jl")
export run_exp

include("results_visual.jl")
export analyze_rewards

end # module
