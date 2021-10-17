using StableRNGs
using LinearAlgebra
using Plots

include("./data_pipeline.jl")

function plot_eigen_spec(m)
      eigen_matrix = eigen(Matrix(m)).vectors

      eigen_imag = imag(eigen_matrix)
      eigen_real = real(eigen_matrix)

      function circShape(h,k,r)
            θ=LinRange(0,2*π, 500)
            h .+ r*sin.(θ), k .+ r*cos.(θ)
      end

      scatter(vec(x_m),vec(y_m), size=(500,500))
      plot!(circShape(0,0,1), seriestype=[:shape,], lw = 0.5,c=:green,legend=false,fillalpha=0.2,aspect_ration=1)
end

lsm = model_dict["RL_LSM"](4,2,"cartpole",StableRNG(123))
plot_eigen_spec(lsm.reservoir.layers[2].W[2])
