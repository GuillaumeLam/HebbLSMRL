# to be put in WaspNet.jl?
mutable struct SpikeTrainGenerator
    d
    sample
    rng

    env_states

    SpikeTrainGenerator(d; visual=false) =
        new(d,[],nothing, visual ? [] : nothing)
    SpikeTrainGenerator(d, rng; visual=false) =
        new(d,[],rng, visual ? [] : nothing)
end

function (stg::SpikeTrainGenerator)(x::AbstractVector)

    if !isnothing(stg.env_states)
        append!(stg.env_states, x)
    end

    if all(x.==0)
        ds = product_distribution(stg.d.(x))
    else
        norm = normalize(x)
        ds = product_distribution(stg.d.(norm))
    end

    if !isnothing(stg.rng)
        stg.sample = rand(stg.rng, ds)
        return (_) -> rand!(stg.rng, ds, stg.sample)
    else
        stg.sample = rand(ds)
        return (_) -> rand!(ds, stg.sample)
    end
end

function (stg::SpikeTrainGenerator)(m::AbstractMatrix)
    return mapslices(stg, m, dims=1)
end
