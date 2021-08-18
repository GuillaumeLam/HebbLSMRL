using ArgParse

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
		"--optimizer", "-o"
			help = "Optimizer for training model"
			arg_type = String
			default = "RMSPROP"
    end

    return parse_args(s)
end

function get_args()
	args = []
	parsed_args = parse_commandline()
	for (arg,val) in parsed_args
		if arg=="model"
			push!(args, val)
		elseif arg=="total-eps"
			if !isnothing(val)
				push!(args, [val])
			else
				push!(args, [100 500])
			end
		end
	end
	return args
end

# model_type, total_eps = get_args()
