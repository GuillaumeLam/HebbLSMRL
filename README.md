# RL-LSM

## PLAN

MVP:

- [ ] get cartpole env working
 - [x] RL.jl => traditional env
 - [ ] MuJoCo.jl => continuous robotic control
- [ ] convert values to spike trains
 Conversion method:
 - [ ] rate based => poisson
 - [ ] sparse coding => time to first spike coding
 - [ ] population encoding => needs critic network...? 
- [ ] make lsm (no stdp)
 - [ ] WaspNet.jl
 - [ ] Flux.jl
- [ ] train and evaluate

ADD-ONs:
- [ ] more complex env
- [ ] implement different encoding methods
- [ ] add stdp
- [ ] different architecture
