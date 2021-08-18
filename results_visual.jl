using DelimitedFiles
using Statistics
using StatsBase
using StatsPlots
using Plots; pyplot()

results_path = "./results/"
save_path = "./plots/"

model_type = "NN"

aggr = Dict("Mean" => Dict(), "IQM" => Dict(), "MEDIAN" => Dict(), "OG" => Dict())

for file in readdir(results_path)
    if !occursin(model_type, file)
        continue
    end

    v = readdlm(results_path*file)

    μ = vec(mean(v, dims=1))
    σ = vec(std(v, dims=1))
    # x̃ = vec(median(v, dims=1))

    display(Plots.plot([1:length(μ);],μ, color=:lightblue, ribbon=σ, label=false))
    # display(Plots.plot([1:length(x̃);],x̃, color=:lightblue, ribbon=σ, label=false))
    savefig(save_path*"Q$(model_type)_avg$(file[(end-6):(end-4)])ep-reward")

    tmp_mean = mean(v,dims=2)
    # MEAN_μ = mean(tmp_mean)
    # MEAN_σ = std(tmp_mean)

    tmp_iq = mapslices((x)->mean(collect(trim(x,prop=0.25))), v, dims=2)
    # IQ_μ = mean(tmp_iq)
    # IQ_σ = std(tmp_iq)

    tmp_med = median(v,dims=2)
    # MED_μ = mean(tmp_med)
    # MED_σ = std(tmp_med)

    γ = 200
    tmp_og = mapslices((x)->mean([min(e,γ) for e in x]), v, dims=2)
    # OG_μ = mean(tmp_og)
    # OG_σ = std(tmp_og)

    aggr["Mean"][file[(end-6):(end-4)]] = tmp_mean
    aggr["IQM"][file[(end-6):(end-4)]] = tmp_iq
    aggr["MEDIAN"][file[(end-6):(end-4)]] = tmp_med
    aggr["OG"][file[(end-6):(end-4)]] = tmp_og
end

function dict_flatten(dict::Dict)
    ks = []
    vs = []

    for (k,v) in dict
        append!(ks, fill(k, length(v)))
        append!(vs, v)
    end

    return ks, vs
end


colors = distinguishable_colors(4, RGB(0.3,0.3,0.4))
plot(layout=(2,2))

for (i, (plot_title, dict)) in enumerate(aggr)
    if i == length(aggr)
        display(boxplot!(dict_flatten(dict)..., color=colors[i], label=false, xlabel=plot_title, subplot=i))
    else
        boxplot!(dict_flatten(dict)..., color=colors[i], label=false, xlabel=plot_title, subplot=i)
    end
end


savefig(save_path*"Q$(model_type)_100vs500eps")
