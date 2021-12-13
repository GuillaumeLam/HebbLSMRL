begin
    using Revise
    using BenchmarkTools
    using StableRNGs
    using HebbLSMRL
end

rng = StableRNG(12345)
v = zeros(100)

HebbLSMRL.run_exp!(rng, v)
# base method 31.416 s (291321671 allocations: 13.41 GiB)
#
# => prealloc result array
# 14.375 s (274835930 allocations: 12.65 GiB)

@code_warntype HebbLSMRL.run_exp!(rng, v)
