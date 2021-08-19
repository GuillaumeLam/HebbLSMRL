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
- [x] make lsm (no stdp)
    - [x] WaspNet.jl
    - [x] Flux.jl
- [x] add train of last layers
- [x] evaluate with RL
- [x] fix gradients

ADD-ONs:
- [ ] more complex env
- [x] implement different input encoding methods
- [ ] add stdp
- [ ] different architecture


WaspNet
- [x] clean up for clean input of poisson neurons to input of sim (make it easier to change prob from sim to sim)
- [x] add unit tests for inhiblif


NOTE:
-until all changes are added to WaspNet.jl, docker is on the backburner
-to fetch plots & results simply run `(sudo) docker cp julia-docker:/app/plots ~/Downloads/plots`
