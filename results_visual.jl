using Statistics
using StatsBase
using Plots; pyplot()

v = readdlm("./results/QLSM-total_ep=100.txt")
# v = rand(10,100)

μ = vec(mean(v, dims=1))
σ = vec(std(v, dims=1))
x̃ = vec(median(v, dims=1))

Plots.plot([1:100;],μ, color=:lightblue, ribbon=σ, label=false)
Plots.plot([1:100;],x̃, color=:lightblue, ribbon=σ, label=false)


tmp_mean = mean(v,dims=2)
MEAN_μ = mean(tmp_mean)
MEAN_σ = std(tmp_mean)

tmp_iq = mapslices((x)->mean(collect(trim(x,prop=0.25))), v, dims=2)
IQ_μ = mean(tmp_iq)
IQ_σ = std(tmp_iq)

tmp_med = median(v,dims=2)
MED_μ = mean(tmp_med)
MED_σ = std(tmp_med)

γ = 200
tmp_og = mapslices((x)->mean([min(e,γ) for e in x]), v, dims=2)
OG_μ = mean(tmp_og)
OG_σ = std(tmp_og)

labels = ["100 ep"]


plot(layout=(2,2))

bar!(labels, [MEAN_μ]; yerr=[MEAN_σ], label=false, bar_edges=true, subplot=1)
plot!(title="MEAN", subplot=1)

bar!(labels, [IQ_μ]; yerr=[IQ_σ], label=false, bar_edges=true, subplot=2)
plot!(title="IQ", subplot=2)

bar!(labels, [MED_μ]; yerr=[MED_σ], label=false, bar_edges=true, subplot=3)
plot!(title="MED", subplot=3)

bar!(labels, [OG_μ]; yerr=[OG_σ], label=false, bar_edges=true, subplot=4)
plot!(title="OG", subplot=4)
