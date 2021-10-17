using WaspNet
using Random
using BenchmarkTools
using Plots
using LinearAlgebra
using SparseArrays

seed = 123
Random.seed!(seed)

inputs = [1,2,2,10]

# layer of neurons

h1 = 8
neurons = [WaspNet.LIF() for _ in 1:h1]
w1 = randn(MersenneTwister(seed), h1, length(inputs))
l1 = Layer(neurons, w1)

update!(l1, poissonST(inputs)(1), 0.001, 0.25)
simulate!(l1, getPoissonST_old(inputs), 0.001, 0.25)


# network of neurons
Nin = 4
N1 = 6
N2 = 4

neurons1 = Vector{AbstractNeuron}([WaspNet.LIF() for _ in 1:5])
append!(neurons1, [WaspNet.InhibNeuron(WaspNet.LIF())])

W1 = randn(MersenneTwister(seed), N1, Nin)
L1 = Layer(neurons1, W1)

neurons2 = [WaspNet.LIF() for _ in 1:N2]
W2 = randn(MersenneTwister(seed), N2, N1)
L2 = Layer(neurons2, W2)

net = Network([L1, L2], N1)

# update!(net, poissonST(inputs)(1), 0.001, 0.250)
# reset!(net)
sim = simulate!(net, poissonST(inputs), 0.001, 1, track_state=true)
# takes 3.109 ms to simulate 1s at 0.1ms timesteps

sum(sim.outputs, dims=2)

spy(sim.outputs[:,1:300])

sim.states[5:10,:]
plot!(sim.states[5:10,:]', layout = (6, 1))

###
function create_conn(val, avg_conn, ub, n)
    if val < avg_conn/n
        return rand()*ub
    else
        return 0.
    end
end

n_in = 10
K = 5

ne = 120
ni = 30
C = 1.

n_lout = 10
n_out = 4

PE_UB = 0.6
EE_UB = 0.05
EI_UB = 0.25
IE_UB = 0.3
II_UB = 0.01

in_n = [WaspNet.LIF() for _ in 1:n_in]
in_w = randn(MersenneTwister(seed), n_in, n_in)
in_w = sparse(in_w)
in_l = Layer(in_n, in_w)

res_n = Vector{AbstractNeuron}([WaspNet.LIF() for _ in 1:ne])
append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF()) for _ in 1:ni])

### res layer weights
W_in = cat(create_conn.(rand(n_in,ne), K, PE_UB, n_in), zeros(n_in,ni), dims=2)'
W_in = sparse(W_in)

W_EI = create_conn.(rand(ne,ni), C, EI_UB, ne)
W_IE = create_conn.(rand(ni,ne), C, IE_UB, ne)
W_EE = W_EI*W_IE
W_EE[diagind(W_EE)] .= 0.
W_II = W_IE*W_EI

W_res = cat(cat(W_EE, W_EI, dims=2),cat(W_IE, W_II, dims=2), dims=1)
W_res = sparse(W_res)

res_w = [W_in, W_res]
conns = [1, 2]

res = Layer(res_n, res_w, conns)

### liquid-out layer weights
lout_n = [WaspNet.ReLU() for _ in 1:n_lout]
W_lout = cat(rand(ne, n_lout), zeros(ni, n_lout), dims=1)'
W_lout = sparse(W_out)

lout_l = Layer(lout_n, W_lout)


out_n = [WaspNet.LIF() for _ in 1:n_out]
W_out = rand(n_out, n_lout)
out_l = Layer(out_n, W_out)

lsm = Network([in_l, res, lout_l, out_l], n_in)

@btime simulate!(lsm, poissonST([1,1,1,1,1,1,1,1,1,1]), 0.001, 0.1, track_state=true)
# takes 29.255 ms to simulate 100ms with res for cartpole
# takes 79.518 ms to simulate 250ms with res for cartpole
# takes 335.399 ms to simulate 1s with res for cartpole

sim = simulate!(lsm, poissonST([1,1,1,1,1,1,1,1,1,1]), 0.001, 0.1, track_state=true)

argmax(sum(sim.outputs[end-3:end,:], dims=2))

for i in 1:5:40
    display(plot(sim.states[i:i+4,:]', layout = (5, 1)))
end

function RL_LSM(NE::AbstractNeuron, NI::AbstractNeuron)

end

function (net::AbstractNetwork)(x::AbstractVector,sim_τ=0.001, sim_T=0.1)
    sim = simulate!(net, poissonST(x), sim_τ, sim_T)

    return normalize(sum(sim.outputs[end-3:end,:],dims=2), sim_T/sim_τ)
end

lsm([1,1,1,1,1,1,1,1,1,1])
