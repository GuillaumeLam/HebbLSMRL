begin
    using DelimitedFiles
    using Statistics
    using StatsBase
    using StatsPlots
    using Plots
    using PyPlot
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

function to_str(vec)
    str = ""

    for e in vec
        str *= string(e)
        str *= "-"
    end

    str = chop(str, tail=1)

    return str
end

function get_top(p,v)

    lim = size(v)[2]
    bot = 0
    if p >= 100
        bot = 1
    elseif p <= 0
        bot = lim
    else
        idx = Int(floor(lim * (1-p/100)))
        bot = idx < 1 ? 1 : idx
    end

    return v[:,bot:end]
end

function analyze_rewards()
    results_path = pwd()*"/results/"
    save_path = pwd()*"/plots/"

    model_type = "LSM"

    aggr = Dict("Mean" => Dict(), "IQM" => Dict(), "MEDIAN" => Dict(), "OG" => Dict())

    isdir(save_path) || mkdir(save_path)

    eps = Vector{Int64}(undef, 0)

    for file in readdir(results_path)
        if !occursin("Q"*model_type, file)
            continue
        end

        v = readdlm(results_path*file)

        # order runs by max val hit, lowest to highest
        v = sortslices(v, dims=2, by=x->max(x...))

        # get top p% runs (0-100)%
        p = 50

        v = get_top(p,v)

        n_eps = size(v)[1]

        μ = vec(mean(v, dims=2))
        σ = vec(std(v, dims=2))
        x̃ = vec(median(v, dims=2))

        # Plots.plot([1:length(μ);],μ, color=:lightblue, ribbon=σ, label=false)
        # Plots.plot!([1:length(x̃);],x̃, color=:lightgreen, ribbon=σ, label=false)
        # Plots.savefig(save_path*"Q$(model_type)_avg&med_e=$(n_eps)")

        display(Plots.plot([1:length(μ);],μ, color=:lightblue, ribbon=σ, label=false))
        Plots.savefig(save_path*"Q$(model_type)_avg_e=$(n_eps)")

        display(Plots.plot([1:length(x̃);],x̃, color=:lightgreen, ribbon=σ, label=false))
        Plots.savefig(save_path*"Q$(model_type)_med_e=$(n_eps)")

        tmp_mean = mean(v,dims=1)
        # MEAN_μ = mean(tmp_mean)
        # MEAN_σ = std(tmp_mean)

        tmp_iq = mapslices((x)->mean(collect(trim(x,prop=0.25))), v, dims=1)
        # IQ_μ = mean(tmp_iq)
        # IQ_σ = std(tmp_iq)

        tmp_med = median(v,dims=1)
        # MED_μ = mean(tmp_med)
        # MED_σ = std(tmp_med)

        γ = 200
        tmp_og = mapslices((x)->mean([min(e,γ) for e in x]), v, dims=1)
        # OG_μ = mean(tmp_og)
        # OG_σ = std(tmp_og)

        append!(eps, n_eps)
        n_eps = string(n_eps)

        aggr["Mean"][n_eps] = tmp_mean
        aggr["IQM"][n_eps] = tmp_iq
        aggr["MEDIAN"][n_eps] = tmp_med
        aggr["OG"][n_eps] = tmp_og
    end

    colors = distinguishable_colors(4, RGB(0.3,0.3,0.4))
    Plots.plot(layout=(2,2))

    for (i, (plot_title, dict)) in enumerate(aggr)
        if i == length(aggr)
            display(boxplot!(dict_flatten(dict)..., color=colors[i], label=false, xlabel=plot_title, subplot=i))
        else
            boxplot!(dict_flatten(dict)..., color=colors[i], label=false, xlabel=plot_title, subplot=i)
        end
    end

    #figure out groupedboxplot

    Plots.savefig(save_path*"Q$(model_type)_[$(to_str(eps))]-eps")
end

analyze_rewards()

#todo
# analyze_rewards() -> analyzes rewards in default path
# analyze_rewards(::String) -> analyzes rewards in given path
# analyze_rewards(::Matrix) -> analyzes rewards of matrix
# analyze_rewards(::Vector) -> make matrix of vector and call above
