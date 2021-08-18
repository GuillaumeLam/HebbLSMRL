using Pkg
Pkg.activate(".")

include("./data_pipeline_util.jl")

using ArgParse
using DelimitedFiles

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--model", "-m"
            help = "Model type to Q-learn"
            arg_type = String
            default = "LSM"
        "--total-eps", "-e"
            help = "Total episodes per experiment"
            arg_type = Int
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
for (arg,val) in parsed_args
	if arg=="model"
		global model_type = val
	elseif arg=="total-eps"
		if !isnothing(val)
			global total_eps = [val]
		else
			global total_eps = [100 500]
		end
	end
end

seeds = [001993 109603 619089 071198 383163 213556 410290 908818 123123 456456]

for (j, total_ep) in enumerate(total_eps)
	@info "Running each experiments for $total_ep episodes"
	for (i, seed) in enumerate(seeds)
		@info "Starting experiment $i"
		reward = run_exp(seed, model_type; total_eps=total_ep)
		@info "Completed $(i/length(seeds)*100)% of experiments of $total_ep episodes"
		io = open("./results/Q$model_type-total_ep=$total_ep.txt", "a") do io
			writedlm(io, reward')
			@info "Logged run!"
		end
	end
	@info "Completed $(j/length(total_eps)*100)% of steps experiments"
end
