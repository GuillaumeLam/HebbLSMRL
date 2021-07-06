using WaspNet
using Random
using BenchmarkTools

seed = 123
Random.seed!(seed)


# layer of neurons
inputs = [1,2,2,10]

h1 = 8
neurons = [WaspNet.LIF() for _ in 1:h1]
w1 = randn(MersenneTwister(seed), h1, length(inputs))
l1 = Layer(neurons, w1)

update!(l1, poissonST(inputs)(1), 0.001, 0.25)
simulate!(l1, poissonST(inputs), 0.001, 0.25)


# network of neurons
Nin = 4
N1 = 6
N2 = 4

neurons1 = [WaspNet.LIF() for _ in 1:N1]
W1 = randn(MersenneTwister(seed), N1, Nin)
L1 = Layer(neurons1, W1)

neurons2 = [WaspNet.LIF() for _ in 1:N2]
W2 = randn(MersenneTwister(seed), N2, N1)
L2 = Layer(neurons2, W2)

net = Network([L1, L2], N1)

update!(net, poissonST(inputs)(1), 0.001, 0.250)
reset!(net)
sim = simulate!(net, poissonST(inputs), 0.001, 1, track_state=true)
# takes 3.109 ms to simulate 1s at 0.1ms timesteps

sum(sim.outputs, dims=2)
