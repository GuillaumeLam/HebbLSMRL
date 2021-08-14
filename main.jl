include("./src/lsm.jl")

using ReinforcementLearning
using Flux
using StableRNGs

seed = 123

rng = StableRNG(seed)
env = CartPoleEnv(; T = Float32, rng = rng)
ns, na = length(state(env)), length(action_space(env))

env_param = LSM.LSMParams(ns*2,na,"cartpole")
W = rand(env_param.n_out,env_param.res_out)
cartpole_lsm = LSM.LSM_Wrapper(env_param, W, rng)
opt = RMSProp(0.0002, 0.99)

policy = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = cartpole_lsm |> cpu,
                optimizer = opt,
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = Flux.mse,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.05,
            decay_steps = 500,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)

stop_condition = StopAfterStep(1000, is_show_progress=!haskey(ENV, "CI"))
hook = TotalRewardPerEpisode()

run(policy, env, stop_condition, hook)
