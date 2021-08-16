include("./src/lsm.jl")

using ReinforcementLearning
using Flux
using StableRNGs

seed = 123

rng = StableRNG(seed)
env = CartPoleEnv(; T = Float32, rng = rng)
ns, na = length(state(env)), length(action_space(env))

env_param = LSM.LSMParams(ns*2,na,"cartpole")
cartpole_lsm = LSM.LSM_Wrapper(env_param, rng)
opt = RMSProp(0.0002, 0.99)
total_steps = 1_000

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
            ϵ_stable = 0.001,
            decay_steps = 0.1*total_steps,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)

stop_condition = StopAfterStep(total_steps, is_show_progress=!haskey(ENV, "CI"))
hook = TotalRewardPerEpisode()

using Zygote
using ReinforcementLearning
using ReinforcementLearning: BasicDQNLearner, NamedTuple
import ReinforcementLearning.RLBase.update!

function RLBase.update!(learner::BasicDQNLearner, batch::NamedTuple{SARTS})

    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_func

    s, a, r, t, s′ = send_to_device(device(Q), batch)
    a = CartesianIndex.(a, 1:length(a))

    gs = gradient(params(Q)) do
        q_tmp = Q(s)
        q = q_tmp[a]
        tmp = Q(s′)

        Zygote.ignore() do
            println("Zero")
            println("Q(s'): $tmp")
        end

        q′ = vec(maximum(tmp; dims = 1))

        Zygote.ignore() do
            println("q: $q \n q′:$(q′)")
        end

        if sum(isnan.(q))>1 || sum(isnan.(q′))>1
            println("we got one BOYSS")
            q = map(x -> isnan(x) ? zero(x) : x, q)
            q′ = map(x -> isnan(x) ? zero(x) : x, q′)
        end

        Zygote.ignore() do
            println("q: $q \n q′:$(q′)")
        end

        G = r .+ γ .* (1 .- t) .* q′

        Zygote.ignore() do
            println("First")
            println("r:$r \n q′:$(q′)")
            println("G: $G \n q: $q")
        end
        loss = loss_func(G, q)
        Zygote.ignore() do
            println("Second")
            learner.loss = loss
            println("Loss: $loss")
        end
        loss
    end

    Zygote.ignore() do
        for (k,v) in gs
           println(any(isnan.(v)))
       end
        println(Q.model.readout_model[1].W)
    end

    update!(Q, gs)
end


cartpole_lsm.readout_model[1].W

run(policy, env, stop_condition, hook)

cartpole_lsm.readout_model[1].W
