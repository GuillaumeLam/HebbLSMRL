using Random, Plots
using DataStructures, RandomNumbers.Xorshifts, StatsBase
import Base.getindex, Base.setindex!

getindex(d::DataStructures.MutableBinaryHeap,i::Int64) = d.nodes[d.node_map[i]].value
setindex!(d::DataStructures.MutableBinaryHeap,v::Complex,i::Int64) = update!(d,i,v)

anim = Plots.Animation()

function spikingnet(n,nspike,k,j0,ratewnt,tau,seedic,seedtopo)

    iext = tau*sqrt(k)*j0*ratewnt/1000
    w, c = 1/log(1. + 1/iext),j0/sqrt(k)/(1. + iext)

    phith, phishift = 1., 0.

    r = Xoroshiro128Star(seedic)

    phi = MutableBinaryMaxHeap(rand(n))

    spikeidx = Int64[]
    spiketimes = Float64[]
    postidx = Array{Int64,1}(undef, k)

    anim = Plots.Animation()

    @time for s=1:nspike
        phimax, j = top_with_handle(phi)
        dphi = phith - (phimax + phishift)
        phishift += dphi
        Random.seed!(r,j+seedtopo)
        sample!(r,1:n-1,postidx)
        @inbounds for i=1:k
            postidx[i] >= j && (postidx[i]+=1)
        end

        ptc!(phi,postidx,phishift,w,c)
        phi[j]=-phishift
        push!(spiketimes,phishift)
        push!(spikeidx,j)

        display(plot(spiketimes*tau/w, spikeidx, seriestype=:scatter, markersize=0.1, legend=false))
        Plots.frame(anim)
    end

    gif(anim, "neurons-sim-n:$n-nspikes:$nspike.gif", fps = 10000)

    nspike/phishift/n/tau*w, spikeidx, spiketimes*tau/w
end

function ptc!(phi,postid, phishift, w, c)
    for i = postid
        phi[i] = -w*log(exp(-(phi[i]+phishift)/w)+c)-phishift
    end
end

n,nspike,k,j0,ratewnt,tau,seedic,seedtopo = 10^5,10^5,50,1,1.,.01,1,1

srate, ssidx, sstimes = spikingnet(100, 100, 10, j0, ratewnt, tau, seedic, seedtopo)

plot(sstimes, ssidx, seriestype=:scatter, legend=false)

rate,sidx,stimes = spikingnet(n,nspike,k,j0,ratewnt,tau,seedic,seedtopo)
@show rate
plot(stimes, sidx, seriestype=:scatter, markersize=0.1, legend=false)
