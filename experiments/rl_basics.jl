using ReinforcementLearning

env = RandomWalk1D()

S = state_space(env)

s = state(env)

A = action_space(env)

run(
   RandomPolicy(),
   RandomWalk1D(),
   StopAfterEpisode(10),
   TotalRewardPerEpisode()
)

NS, NA = length(S), length(A)

policy = TabularPolicy(;table=Dict(zip(1:NS, fill(NA, NS))),n_action=NA)

run(
  policy,
  RandomWalk1D(),
  StopAfterEpisode(10),
  TotalRewardPerEpisode()
)

using Flux: InvDecay

policy = QBasedPolicy(
  learner = MonteCarloLearner(
    ;approximator=TabularQApproximator(
      ;n_state = NS,
      n_action = NA,
      opt = InvDecay(1.0)
    )
  ),
  explorer = EpsilonGreedyExplorer(0.1)
)

run(
   policy,
   RandomWalk1D(),
   StopAfterEpisode(10),
   TotalRewardPerEpisode()
)

agent = Agent(
  policy = policy,
  trajectory = VectorSARTTrajectory()
)

run(agent, env, StopAfterEpisode(10), TotalRewardPerEpisode())

cp_agent = Agent(
  policy = cp_policy,
  trajectory = VectorSARTTrajectory()
)

run(agent, CartPoleEnv(), StopAfterEpisode(10), TotalRewardPerEpisode() )
