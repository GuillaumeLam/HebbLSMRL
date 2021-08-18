include("./src/lsm.jl")

using ReinforcementLearning
using Flux
using Random
using StableRNGs

cartpole_lsm(ns, na, env, rng) = begin
    env_param = LSM.LSMParams(ns*2,na,"cartpole")
    LSM.LSM_Wrapper(env_param, rng, (x)->(LSM.genPositive(LSM.genCapped(x,[2.5,0.5,0.28,0.88]))))
end

cartpole_lsm_discr(ns, na, env, rng) = begin
    n = 10
    env_param = LSM.LSMParams(ns*(n+1),na,"cartpole")
    LSM.LSM_Wrapper(env_param, rng, (x)->LSM.discretize(x,[2.5,0.5,0.28,0.88], n))
end

cartpole_nn(ns, na, env, rng) = begin
    Chain(
        Dense(ns, 128, relu; init = glorot_uniform(rng)),
        Dense(128, 128, relu; init = glorot_uniform(rng)),
        Dense(128, na; init = glorot_uniform(rng)),
    )
end

model_dict = Dict(
    "LSM" => cartpole_lsm,
    "DLSM" => cartpole_lsm_discr,
    "NN" => cartpole_nn,
    # "L-STDP" => () -> (println("to be implemented!");throw MethodError),
    )

opt_dict = Dict(
    "ADAM" => ADAM(0.01),
    "RMSPROP" => RMSProp(0.0002, 0.99),
)

function run_exp(seed, model_name::String="LSM"; total_eps=100)
    rng = StableRNG(seed)
    Random.seed!(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    model = model_dict[model_name](ns, na, "cartpole", rng)

    opt = opt_dict["RMSPROP"]

    total_steps = 1000

    policy = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = model |> cpu,
                    optimizer = opt,
                ),
                batch_size = 32,
                min_replay_history = Int64(0.1*total_steps),
                loss_func = Flux.mse,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.001,
                decay_steps = 100,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = Int64(0.1*total_steps),
            state = Vector{Float32} => (ns,),
        ),
    )

    # stop_condition = StopAfterStep(total_steps, is_show_progress=!haskey(ENV, "CI"))
    stop_condition = StopAfterEpisode(total_eps, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    run(policy, env, stop_condition, hook)

    return hook.rewards
end
