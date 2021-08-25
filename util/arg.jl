using ArgParse

mutable struct Args
	model
	total_eps
	num_of_simulations

	Args() = new("LSM", [100 500 1000 10_000] ,1)

	function (a::Args)(param, val)
		try
			setproperty!(a, Symbol(param), val)
		catch
			println("Unknow argument parameter")
			throw(ErrorException)
		end
	end
end

get_main_arg(a::Args) = return a.model, a.total_eps
get_t_main_arg(a::Args) = return a.model, a.total_eps, a.num_of_simulations

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--model", "-m"
            help = "Model type to train with Q-learning"
            arg_type = String
        "--total_eps", "-e"
            help = "Total episodes per experiment"
            arg_type = Int
		"--num_of_simulations", "-s"
			help = "Total number of simulations to run.
					It will be rounded up to account for
					all threads in the multi-threaded main."
			arg_type = Int
		# "--optimizer", "-o"
		# 	help = "Optimizer for training model"
		# 	arg_type = String
		# 	default = "RMSPROP"
    end

    return parse_args(s)
end

function get_Args()
	args = Args()
	parsed_args = parse_commandline()
	for (arg,val) in parsed_args
		if val != nothing
			args(arg, val)
		end
	end
	return args
end

# model_type, total_eps, #_sim = get_args()
