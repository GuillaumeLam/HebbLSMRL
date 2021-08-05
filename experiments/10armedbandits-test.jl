using ReinforcementLearning
using Flux

env = MultiArmBanditsEnv()

ns, na = length(state(env)), length(action_space(env))

policy = QBasedPolicy(
    learner = MonteCarloLearner(
        approximator = TabularQApproximator(
            n_state = ns,
            n_action = na,
            opt = InvDecay(1.0)
        )
    ),
    explorer = EpsilonGreedyExplorer(0.1)
)

trajectory = VectorSARTTrajectory()

agent = Agent(
    policy = policy,
    trajectory = trajectory
)

run(agent, env, StopAfterStep(100), TotalRewardPerEpisode())
