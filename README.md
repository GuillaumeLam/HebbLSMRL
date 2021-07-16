# RL-LSM

## PLAN

MVP:

- [x] get cartpole env working
    - [x] RL.jl => traditional env
    - [ ] MuJoCo.jl => continuous robotic control
- [x] convert values to spike trains
    Conversion method:
    - [x] rate based => poisson
    - [ ] sparse coding => time to first spike coding
    - [ ] population encoding => needs critic network...?
    - [ ] Dont convert and use cnn
- [ ] make lsm (no stdp)
    - [x] WaspNet.jl
    - [ ] Flux.jl
- [ ] add train of last layers
- [ ] evaluate with RL

ADD-ONs:
- [ ] more complex env
- [ ] implement different input encoding methods
- [ ] add stdp
- [ ] different architecture
WaspNet
 - [x] clean up for clean input of poisson neurons to input of sim (make it easier to change prob from sim to sim)
