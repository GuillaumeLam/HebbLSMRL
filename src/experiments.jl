using ReinforcementLearning
using Flux
using Random

using LiquidStateMachine

using BenchmarkTools

cartpole_lsm(ns, na, rng) = begin
    # env_param = LSM_Params(ns*2,na,"cartpole")
    # return LSM(env_param, (x)->(genPositive(x)), rng=rng, visual=true)
    return LSM(LSM_Params(ns*2,na,"cartpole"), (x)->(genPositive(x)), rng=rng)
    # return LSM(env_param, rng, (x)->(genPositive(genCapped(x,[2.5,0.5,0.28,0.88]))); visual=true)
end

discr_cartpole_lsm(ns, na, rng) = begin
    # n = 10
    # env_param = LSM_Params(ns*(n+1),na,"cartpole")
    # return LSM(env_param, (x)->discretize(x,[2.5,0.5,0.28,0.88], n), rng=rng)
    return LSM(LSM_Params(ns*(10+1),na,"cartpole"), (x)->discretize(x,[2.5,0.5,0.28,0.88], 10), rng=rng)
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
    "ADAM" => ADAM(1e-3), #ADAM(1e-3)
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

function get_agent(rng, env, model_type, opt_type, total_eps, update_freq)
    ns, na = length(state(env)), length(action_space(env))
    # 1.520 ns (0 allocations: 0 bytes)

    model = model_dict[model_type](ns, na, rng)
    # 201.675 μs (372 allocations: 979.58 KiB)

    opt = opt_dict[opt_type]
    # 27.657 ns (0 allocations: 0 bytes)

    total_steps = total_eps*100
    # 0.018 ns (0 allocations: 0 bytes)

    rand_agent = RandomPolicy(action_space(env))
    # 1.273 ns (0 allocations: 0 bytes)

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
    # 6.682 μs (67 allocations: 3.44 KiB)

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

#todo
# - make function iter over rngs and place subsequent in out_m
# - throw err if length(out_v)≠total_eps
# - run_exp!(::Vector{RNG}, <:AbstractArrays) -> make sure both SharedArray & Matrix work
#     run_exp_v!(::RNG, ::Vector) (version for one rng)
# - use @allocated to track actual val or use TimerOutputs
# - Profile LSM and clean up a bit
function run_exp!(rng, out_v; model_type::String="LSM", total_eps=100, visual=nothing)
    # rng = StableRNG(seed)
    # Random.seed!(seed)

    env = CartPoleEnv(T=Float32, rng=rng)
    # 2.361 μs (33 allocations: 1.06 KiB)

    agent = get_agent(rng, env, model_type, "RMSPROP", total_eps, 10)
    # 416.113 μs (439 allocations: 983.02 KiB)
    # => packed in 1 line LSM init
    # 224.813 μs (437 allocations: 982.94 KiB)
    # => no visual
    # 222.399 μs (417 allocations: 980.67 KiB)


    # stop_condition = StopAfterStep(total_steps, is_show_progress=!haskey(ENV, "CI"))
    stop_condition = StopAfterEpisode(total_eps, is_show_progress=!haskey(ENV, "CI"))
    # 706.248 ns (9 allocations: 576 bytes)
    hook = TotalRewardPerEpisode()
    # 26.175 ns (2 allocations: 112 bytes)

    run(agent, env, stop_condition, hook)
    # 30.661 ms (229412 allocations: 10.64 MiB)

    # println(model.readout.layers[1].W)

    # fetch exp run states
    # println(Q_agent.policy.learner.approximator.model.states_dict) # => returning nothing for some reason

    if !isnothing(visual) && isa(visual, Vector{AbstractDict})
        # frames = Q_agent.policy.learner.approximator.model.states_dict
        # push!(visual, frames)
        push!(visual, Q_agent.policy.learner.approximator.model.states_dict)
    end
    # 0.017 ns (0 allocations: 0 bytes)

    # println(size(frames["env"]))
    # println(size(frames["out"]))
    # println(size(frames["spike"]))
    # println(size(frames["spike"][1]))

    copy!(out_v, hook.rewards)
    # 17.770 ns (0 allocations: 0 bytes)

    GC.gc()
end
