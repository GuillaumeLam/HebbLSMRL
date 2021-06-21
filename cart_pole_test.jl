using ReinforcementLearning
using Flux
using Flux.Losses
using StableRNGs

seed = 666
rng = StableRNG(seed)
env = CartPoleEnv(; T = Float32, rng = rng)
ns, na = length(state(env)), length(action_space(env))

policy = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu),
                    Dense(128, 128, relu),
                    Dense(128, 128, relu),
                    Dense(128, na, relu)
                ),
                optimizer = ADAM()
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)


run(policy, env, StopAfterStep(2000), TotalRewardPerEpisode())
