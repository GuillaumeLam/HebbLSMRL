@with_kw struct LIFC{T<:Number}<:AbstractNeuron 
    τ::T = 8.
    R::T = 10.
    θ::T = -55.

    vSS::T = -50.
    v0::T = -100.
    state::T = -100.
    output::T = 0.
end

function update(neuron::LIFC, input_update, dt, t)
    output = 0.
    # If an impulse came in, add it
    state = neuron.state + input_update * neuron.R / neuron.τ

    # Euler method update
    state += 1000 * (dt/neuron.τ) * (-state + neuron.vSS)

    # Check for thresholding
    if state >= neuron.θ
        state = neuron.v0
        output = 1. # Binary output
    end

    return (output, LIF(neuron.τ, neuron.R, neuron.θ, neuron.vSS, neuron.v0, state, output))
end

function reset(neuron::LIFC)
    return LIF(neuron.τ, neuron.R, neuron.θ, neuron.vSS, neuron.v0, neuron.v0, 0.)
end
