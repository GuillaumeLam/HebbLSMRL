include("../src/lsm.jl")

using ReinforcementLearning
using Flux
using Random

cartpole_lsm(ns, na, rng) = begin
    env_param = RL_LSM.LSM_Params(ns*2,na,"cartpole")
    RL_LSM.LSM(env_param, rng, (x)->(RL_LSM.genPositive(RL_LSM.genCapped(x,[2.5,0.5,0.28,0.88]))))
end

cartpole_lsm_discr(ns, na, rng) = begin
    n = 10
    env_param = RL_LSM.LSM_Params(ns*(n+1),na,"cartpole")
    RL_LSM.LSM(env_param, rng, (x)->RL_LSM.discretize(x,[2.5,0.5,0.28,0.88], n))
end

cartpole_nn(ns, na, rng) = begin
    Chain(
        Dense(ns, 128, relu; init = glorot_uniform(rng)),
        Dense(128, 128, relu; init = glorot_uniform(rng)),
        Dense(128, na; init = glorot_uniform(rng)),
    )
end

model_dict = Dict(
    "RL_LSM" => cartpole_lsm,
    "DRL_LSM" => cartpole_lsm_discr,
    "NN" => cartpole_nn,
    # "L-STDP" => () -> (println("to be implemented!");throw MethodError),
    )

opt_dict = Dict(
    "ADAM" => ADAM(0.01),
    "RMSPROP" => RMSProp(0.0002, 0.99),
)

#=
    make dict of td method for rl

rl_dict = Dict(
    "QL" =>
    "AC" =>
)

=#

function run_exp(rng, model_name::String="RL_LSM"; total_eps=100, visual=false)
    # rng = StableRNG(seed)
    # Random.seed!(seed)

    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    model = model_dict[model_name](ns, na, rng)

    opt = opt_dict["RMSPROP"]

    total_steps = total_eps*100

    policy = Agent(
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
            capacity = Int64(0.1*total_steps),
            state = Vector{Float32} => (ns,),
        ),
    )

    # stop_condition = StopAfterStep(total_steps, is_show_progress=!haskey(ENV, "CI"))
    stop_condition = StopAfterEpisode(total_eps, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()

    run(RandomPolicy(action_space(env)), env, stop_condition, hook)

    # println(model.readout.layers[1].W)

    return hook.rewards
end
