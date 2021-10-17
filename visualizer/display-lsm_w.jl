struct CartPoleVisualizer
    lsm_w::LSM

    function (v::CartPoleVisualizer)(x)
        #plot x ie cartpole system
        z = v.lsm_w(x)
        #plot z, the outgoing st
        return z
    end
end
