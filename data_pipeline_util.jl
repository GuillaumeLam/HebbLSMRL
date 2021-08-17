include("./src/lsm.jl")

using ReinforcementLearning
using Flux
using Random
using StableRNGs

function run_exp(seed; total_eps=100)
    rng = StableRNG(seed)
    Random.seed!(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    env_param = LSM.LSMParams(ns*2,na,"cartpole")
    cartpole_lsm = LSM.LSM_Wrapper(env_param, rng)
    opt = RMSProp(0.0002, 0.99)
    # opt = ADAM(0.01)
    total_steps = 1000

    policy = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = cartpole_lsm |> cpu,
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
