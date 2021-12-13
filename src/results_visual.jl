begin
    using DelimitedFiles
    using Statistics
    using StatsBase
    using StatsPlots
    using Plots
    # using PyPlot
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

    # remove trailing dash
    str = chop(str, tail=1)

    return str
end

function get_top(p,m)
    lim = size(m)[2]
    bot = 0

    if p >= 100
        bot = 1
    elseif p <= 0
        bot = lim
    else
        idx = Int(floor(lim * (1-p/100)))
        bot = idx < 1 ? 1 : idx
    end

    return m[:,bot:end]
end

struct AggrMetric
    dict::Dict

    function AggrMetric(metrics::AbstractVector)
        d = Dict()
        for m in metrics
            d[m] = Dict()
        end
        return new(d)
    end

    AggrMetric() = AggrMetric(["MEAN", "IQM", "MEDIAN", "BASE"])
end

function add!(aggr::AggrMetric, metric, ep, val)
    aggr.dict[metric][ep] = val
end


function analyze_run!(m::AbstractMatrix, aggr=nothing::Union{AggrMetric,Nothing}; top_p=nothing::Union{Int64,Nothing})
    # order the runs by max val hit, lowest to highest in cols
    m = sortslices(m, dims=2, by=x->max(x...))

    # get top p% runs (0-100)%
    if !isnothing(top_p)
        m = get_top(top_p, m)
    end

    μ = vec(mean(m, dims=2))
    σ = vec(std(m, dims=2))
    x̃ = vec(median(m, dims=2))

    p1 = Plots.plot(
        [1:length(μ);], μ, ribbon=σ,
        color=:lightblue,
        xlabel="Episode",
        ylabel="Reward",
        title="Average reward over episode",
        label=false)
    p2 = Plots.plot(
        [1:length(x̃);], x̃, ribbon=σ,
        color=:lightgreen,
        xlabel="Episode",
        ylabel="Reward",
        title="Median reward over episode",
        label=false)
    display(Plots.plot(p1, p2, layout=(2,1), plot_title=(isnothing(top_p) ? "All Runs" : "Top $top_p% Runs")))

    if !isnothing(aggr)
        tmp_mean = mean(m,dims=1)
        # MEAN_μ = mean(tmp_mean)
        # MEAN_σ = std(tmp_mean)

        tmp_iq = mapslices((x)->mean(collect(StatsBase.trim(x,prop=0.25))), m, dims=1)
        # IQ_μ = mean(tmp_iq)
        # IQ_σ = std(tmp_iq)

        tmp_med = median(m,dims=1)
        # MED_μ = mean(tmp_med)
        # MED_σ = std(tmp_med)

        γ = 200
        tmp_base = mapslices((x)->mean([min(e,γ) for e in x]), m, dims=1)
        # base_μ = mean(tmp_base)
        # base_σ = std(tmp_base)

        # pad to preserve order when num of episode gets large
        n_eps = string(size(m)[1])
        padded_n_eps = lpad.(n_eps, 5)
        add!(aggr, "MEAN", padded_n_eps, tmp_mean)
        add!(aggr, "IQM", n_eps, tmp_iq)
        add!(aggr, "MEDIAN", n_eps, tmp_med)
        add!(aggr, "BASE", n_eps, tmp_base)
    end
end

function analyze_aggregate(model_type="LSM"::String; top_p=nothing::Union{Int64,Nothing})
    results_path = pwd()*"/results/"
    save_path = pwd()*"/plots/"

    aggr = AggrMetric()

    isdir(save_path) || mkdir(save_path)

    for file in readdir(results_path)
        if !occursin("Q"*model_type, file)
            continue
        end

        m = readdlm(results_path*file)
        analyze_run!(m, aggr, top_p=top_p)
        n_eps = size(m)[1]
        Plots.savefig(save_path*"Q$(model_type)_avg&med_top=$(isnothing(top_p) ? "100" : string(top_p))_e=$(n_eps)")
    end

    colors = distinguishable_colors(4, RGB(0.3,0.3,0.4))
    Plots.plot(layout=(2,2))

    for (i, (plot_title, dict)) in enumerate(aggr.dict)
        if i == length(aggr.dict)
            display(boxplot!(dict_flatten(dict)..., color=colors[i], label=false, xlabel=plot_title, subplot=i))
        else
            boxplot!(dict_flatten(dict)..., color=colors[i], label=false, xlabel=plot_title, subplot=i)
        end
    end

    Plots.plot!(plot_title=(isnothing(top_p) ? "All Runs" : "Top $top_p% Runs"))

    #figure out groupedboxplot

    eps = sort(collect(keys(aggr.dict["IQM"])))
    Plots.savefig(save_path*"Q$(model_type)_top=$(isnothing(top_p) ? "100" : string(top_p))_e=[$(to_str(eps))]")
end
