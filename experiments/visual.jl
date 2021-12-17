using HebbLSMRL
using StableRNGs

m_seed = 161803
# m_seed = 314156
m_rng = StableRNG(m_seed)

seeds = rand(m_rng, 000000:999999, 2)
rngs = StableRNG.(seeds)

loss_vec = HebbLSMRL.exp(rngs, 100, visual=true)
