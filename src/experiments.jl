cartpole_lsm(ns, na, rng) = begin
    return LSM(LSM_Params(ns*2,na,"cartpole"), (x)->(genPositive(x)), rng=rng)
    # return LSM(env_param, rng, (x)->(genPositive(genCapped(x,[2.5,0.5,0.28,0.88]))); visual=true)
end

discr_cartpole_lsm(ns, na, rng) = begin
    n = 10
    return LSM(LSM_Params(ns*(n+1),na,"cartpole"), (x)->discretize(x,[2.5,0.5,0.28,0.88], n), rng=rng)
end

cartpole_nn(ns, na, rng) = begin
    return Chain(
        Dense(ns, 128, relu; init = glorot_uniform(rng)),
        Dense(128, 128, relu; init = glorot_uniform(rng)),
        Dense(128, na; init = glorot_uniform(rng)),
    )
end

model_dict = Dict(
    "LSM" => cartpole_lsm,
    "DLSM" => discr_cartpole_lsm,
    "NN" => cartpole_nn,
    # "L-STDP" => () -> (println("to be implemented!");throw MethodError),
    )

opt_dict = Dict(
    "ADAM" => ADAM(1e-3),
    "RMSPROP" => RMSProp(0.0002, 0.99),
)

#=
    make dict of td method for rl

rl_dict = Dict(
    "QL" =>
    "AC" =>
    "RND" =>
)

=#

function RLBase.update!(learner::BasicDQNLearner, batch::NamedTuple{SARTS})

    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_func

	println("hello")

    s, a, r, t, s′ = send_to_device(device(Q), batch)
    a = CartesianIndex.(a, 1:length(a))

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        q′ = vec(maximum(Q(s′); dims = 1))
        G = @. r + γ * (1 - t) * q′
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end

function get_agent(rng, env, model_type, opt_type, total_eps, update_freq)
    ns, na = length(state(env)), length(action_space(env))

    model = model_dict[model_type](ns, na, rng)

    opt = opt_dict[opt_type]

    total_steps = total_eps*100

    rand_agent = RandomPolicy(action_space(env))

    return Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = model |> cpu,
                    optimizer = opt,
                ),
                batch_size = 1,
                min_replay_history = Int64(0.1*total_steps),
                loss_func = Flux.mse,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.001,
                decay_steps = Int64(0.1*total_steps),
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = update_freq,
            # capacity = Int64(0.1*total_steps),
            state = Vector{Float32} => (ns,),
        ),
    )

    #=
    A2C_agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
                approximator = ActorCritic(
                    actor = model,
                    critic = cartpole_lsm_discr(ns, 1, rng),
                    optimizer = opt,
                ) |> cpu,
                γ = 0.99f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Vector{Float32} => (ns,),
        ),
    )
    =#
end

function exp_base(rng; model_type::String="LSM", total_eps::Int=100, loss_vec=nothing)
    # rng = StableRNG(seed)
    # Random.seed!(seed)

    env = CartPoleEnv(T=Float32, rng=rng)

    agent = get_agent(rng, env, model_type, "RMSPROP", total_eps, 10)

    # stop_condition = StopAfterStep(total_steps, is_show_progress=!haskey(ENV, "CI"))
    stop_condition = StopAfterEpisode(total_eps, is_show_progress=!haskey(ENV, "CI"))
    # hook = TotalRewardPerEpisode()

	# losses = Vector{Float64}()

	hook = ComposedHook(
		TotalRewardPerEpisode(),
		#make custom hook to contain losses?
		DoEveryNEpisode() do t, agent, env
			# println(t, agent, env)
			# println(agent.policy.learner.loss)
			!isnothing(loss_vec) ? push!(loss_vec, agent.policy.learner.loss) : nothing
		end
	)

    run(agent, env, stop_condition, hook)

    # println(model.readout.layers[1].W)

    # fetch exp run states
    # println(Q_agent.policy.learner.approximator.model.states_dict) # => returning nothing for some reason

    # if !isnothing(visual) && isa(visual, Vector{AbstractDict})
    #     # frames = Q_agent.policy.learner.approximator.model.states_dict
    #     # push!(visual, frames)
    #     push!(visual, Q_agent.policy.learner.approximator.model.states_dict)
    # end

    # println(size(frames["env"]))
    # println(size(frames["out"]))
    # println(size(frames["spike"]))
    # println(size(frames["spike"][1]))

    return hook
end

function run_exp!(rng, out_v; model_type::String="LSM", total_eps::Int=100, loss_vec=nothing)
	hook = exp_base(rng, model_type=model_type, total_eps=total_eps, loss_vec=loss_vec)
    copy!(out_v, hook[1].rewards)
end

function run_exp(rng; model_type::String="LSM", total_eps::Int=100, loss_vec=nothing)
    exp_base(rng, model_type=model_type, total_eps=total_eps, loss_vec=loss_vec)
end

function exp(rngs::AbstractVector{R}, total_eps; parallel=false, model_type::String="LSM", visual=false) where {R<:AbstractRNG}
	loss_vec = Vector{Any}()

	for (j, total_ep) in enumerate(total_eps)
		@info "Running each experiments for $total_ep episodes"
		isdir("../results") || mkdir("../results")

		ep_loss_vec = Vector{Any}()

		if visual
			for _ in 1:length(rngs)
				push!(ep_loss_vec, Vector{Float64}())
			end
		end

		if !parallel
			frame = Matrix{Float64}(undef, total_ep, length(rngs))

			for (i, rng) in enumerate(rngs)
				@info "Starting experiment $i"

				if visual
					run_exp!(rng, frame[:,i], model_type=model_type, total_eps=total_ep, loss_vec=ep_loss_vec[i])
				else
					run_exp!(rng, frame[:,i], model_type=model_type, total_eps=total_ep)
				end

				@info "Completed $(i/length(rngs)*100)% of experiments of $total_ep episodes"
			end

			# store col first
			io = open("../results/Q$model_type-e=$total_ep.txt", "a") do io
				writedlm(io, frame)
				@info "Logged all seeded experiments for $total_ep episodes!"
			end
		else
			@info "Launching parallel exp"

			rewards = pmap((rng)->(run_exp(rng, model_type=model_type, total_eps=total_ep)), rngs)
			# store col first
			io = open("../results/Q$model_type-e=$total_ep.txt", "w") do io
				writedlm(io, hcat(rewards...))
				@info "Logged all seeded experiments for $total_ep episodes!"
			end
		end

		@info "Completed $(j/length(total_eps)*100)% of steps experiments"

		push!(loss_vec, ep_loss_vec)
	end

	if visual
		return loss_vec
	end
end
