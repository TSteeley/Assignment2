using Plots, CustomPlots, JLD2, Distributions, LinearAlgebra, LaTeXStrings
using KernelDensity, Flux, CUDA, ProgressBars
include("../myTheme.jl")
include("../int.jl")

@load "savedData/true.jld2" θt yt
@load "savedData/InferenceSamples.jld2" θi yi

N = round.(Int, 500*(25).^([0:0.1:1.0;]))

# ================================================
# ============    ABC Convergence    =============
# ================================================

ϵ_ABC = mapslices(x -> norm(yt - x), yi, dims = 1)[:]

μ_ABC = zeros(11)
σ_ABC = zeros(11)
for (i,n) ∈ enumerate(N)
    idx = sortperm(ϵ_ABC[1:n])[1:500]
    μ_ABC[i] = mean(θi[idx])
    σ_ABC[i] = 1.96*std(θi[idx])/sqrt(500)
end

# ==================================================
# ======    Sufficient Statistic Convergence    ====
# ==================================================


t_SS =  (sum(yt[1:end-1])-sum(yt[2:end])) / sum(yt[2:end])
SS = mapslices(y -> (sum(y[1:end-1])-sum(y[2:end])) / sum(y[2:end]), yi, dims = 1)[:]

ϵ_SS = map(x -> norm(t_SS - x), SS)

μ_SS = zeros(11)
σ_SS = zeros(11)
for (i,n) ∈ enumerate(N)
    idx = sortperm(ϵ_SS[1:n])[1:500]
    μ_SS[i] = mean(θi[idx])
    σ_SS[i] = 1.96*std(θi[idx])/sqrt(500)
end

plot(N, σ_SS, yerr = 1.96*σ_SS, yscale=:log10, xscale=:log10)

# ================================================
# ============    PCA Convergence    =============
# ================================================

@load "savedData/PCATransform.jld2" PCA_Transform

st = PCA_Transform' * yt[2:end]
si = (yi[2:end,:]' * PCA_Transform)'
ϵ_PCA = mapslices(x -> norm(st - x), si, dims = 1)[:]

μ_PCA = zeros(11)
σ_PCA = zeros(11)
for (i,n) ∈ enumerate(N)
    idx = sortperm(ϵ_PCA[1:n])[1:500]
    μ_PCA[i] = mean(θi[idx])
    σ_PCA[i] = 1.96*std(θi[idx])/sqrt(500)
end

plot(N, σ_PCA, yerr = 1.96*σ_PCA, yscale=:log10, xscale=:log10)

# ================================================
# ========    Autoencoder Convergence    =========
# ================================================

@load "savedData/MyModel.jld2" model_state
include("../models/Autoencoder.jl")
m = INCA(5,1)
m = Flux.loadmodel!(m, model_state)

# Due to RAM constraints this step must be performed in chunks
siAE = zeros(5, 10_000_000)
n = 200_000
for i in 1:50 |> ProgressBar
    @inbounds siAE[:,(i-1)*n+1:i*n] = m.encoder(reshape(yi[:,(i-1)*n+1:i*n], size(yi[:,(i-1)*n+1:i*n],1), 1, :))
end
stAE = m.encoder(reshape(yt, length(yt), 1, :))[:]

ϵ_AE = mapslices(x -> norm(stAE - x), siAE, dims = 1)[:]

μ_AE = zeros(11)
σ_AE = zeros(11)
for (i,n) ∈ enumerate(N)
    idx = sortperm(ϵ_AE[1:n])[1:500]
    μ_AE[i] = mean(θi[idx])
    σ_AE[i] = 1.96*std(θi[idx])/sqrt(500)
end

# ================================================
# ===========    Plot Convergence    =============
# ================================================


p = plot(xlabel = "Samples", ylabel="k",xscale=:log10)
plot!(p,N, μ_ABC, yerror = σ_ABC, msw=2,label="ABC")
plot!(p,N, μ_SS, yerror = σ_SS, msw=2,label="SS")
plot!(p,N, μ_PCA, yerror = σ_PCA, msw=2,label="PCA")
plot!(p,N, μ_AE, yerror = σ_AE, msw=2,label="AE")
hline!(p, [0.25], label = "k true", lc = :black)

savefig(p, "../Assignment2Writing/figures/convergence.pdf")