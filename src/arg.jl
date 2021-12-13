using ArgParse

mutable struct Args
	model
	total_eps
	num_of_simulations
	parallel

	Args() = new("LSM", [100 250 500 1000], 32, true)

	function (a::Args)(param, val)
		try
			setproperty!(a, Symbol(param), val)
		catch
			println("Unknow argument parameter")
			throw(ErrorException)
		end
	end
end

args(a::Args) = return a.model, a.total_eps, a.num_of_simulations, a.parallel

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
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
		"--parallel", "-p"
			help= "Run simulations in parallel?"
			arg_type = Bool
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
