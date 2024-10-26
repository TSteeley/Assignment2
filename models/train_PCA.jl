using JLD2, LinearAlgebra, Distributions, Plots
include("../myTheme.jl")

@load "SavedData/PriorPredictiveSamples.jld2" θp yp

Eyp = mean(yp[2:end,:], dims = 2)[:]
Vyp = var(yp[2:end,:], dims = 2)[:]
nyp = (yp[2:end,:] .- Eyp) ./ sqrt.(Vyp)

covMat = cov(nyp, dims = 2)
eigVals, eigVecs = eigen(covMat, sortby = x -> -x)

# Find the first dimension such that at least 99% of the variance in explained
dims = findfirst(cumsum(eigVals) / sum(eigVals) .> 0.99)

p = plot(
    0:50, vcat(0, cumsum(eigVals) / sum(eigVals)),
    xlabel = "Dimensions", ylabel = "% of Explained Variance",
    label = ""
)

savefig(p, "../Assignment2Writing/figures/PCA_dims.pdf")

PCA_Transform = eigVecs[:,1:dims]

@save "savedData/PcaTransform.jld2" PCA_Transform