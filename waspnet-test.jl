using WaspNet
using LinearAlgebra
using Random, Distributions
using BenchmarkTools

seed = 123
Random.seed!(seed)

# single neuron
n = WaspNet.PoissonLIF(p=0.5)
_, n2 = update(n, 0.001, 0.)

WaspNet.get_neuron_outputs(n2)
simulate!(n, (t)->t, 0.0001, 0.250)


# layer of neurons
inputs = [30, 4, 45, 10]

neurons = [WaspNet.PoissonLIF(prob, -100., -100., 0.) for prob in normalize(inputs)]
weights = [1.0 1 1 1]

layer = Layer(neurons, weights)

update!(layer, [1.0, 2.0 ,3.0 ,4.0], 0.001, 0.01)

simulate!(layer, (t)->t, 0.001, 0.01)


# network of neurons
Nin = 4
N1 = 16
N2 = 4

neuronsin = [WaspNet.Poisson(p=prob) for prob in normalize(inputs)]
Win = [1.0 1 1 1;
        1 1 1 1;
        1 1 1 1;
        1 1 1 1]
Lin = Layer(neuronsin, Win)

neurons1 = [WaspNet.LIF() for _ in 1:N1]
W1 = randn(MersenneTwister(seed), N1, Nin)
L1 = Layer(neurons1, W1)

neurons2 = [WaspNet.LIF() for _ in 1:N2]
W2 = randn(MersenneTwister(seed), N2, N1)
L2 = Layer(neurons2, W2)

net = Network([Lin, L1, L2], Nin)

reset!(net)
update!(net, ones(4), 0.001, 0.250)
@btime sim = simulate!(net, (t)->zeros(4), 0.001, 1, track_state=true)
# takes 7.867 ms to simulate 1s at 0.1ms timesteps

sim.outputs[end-N2+1:end,:]
