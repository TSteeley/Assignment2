using Flux, CUDA, Statistics, ProgressBars, Plots, JLD2, CustomPlots
using Random
include("Autoencoder.jl")

@load "savedData/PriorPredictiveSamples.jld2" θp yp

# Reshape data for use
X = Float32.(yp[:,1:10_000]) |> gpu
θ = Float32.(θp[1:10_000]) |> gpu

model = INCA(5, 1) |> gpu

opt = Flux.setup(Flux.Adam(), model)
idx = randperm(length(θ))

train = Flux.DataLoader((X[:,idx[1:floor(Int,0.7*length(θ))]], θ[idx[1:floor(Int,0.7*length(θ))]]') |> gpu, batchsize = 32)
val = (X[:,idx[ceil(Int,0.7*length(θ)):end]], θ[idx[ceil(Int,0.7*length(θ)):end]]')

for epoch in 1:100
    println("Epoch $(epoch) commencing...")
    for (i,d) in enumerate(train) |> ProgressBar
        x, y = d
        tLoss, grads = Flux.withgradient(model) do m
            ŷ = m(x)
            LossINCA(ŷ, y)
        end

        Flux.update!(opt, model, grads[1])
    end
end

model = cpu(model)
model_state = Flux.state(model)

@save "savedData/MyModel.jld2" model_state