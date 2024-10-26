using Plots, CustomPlots, JLD2, Distributions, LinearAlgebra, LaTeXStrings
using KernelDensity, Flux, CUDA, ProgressBars
include("../myTheme.jl")
include("../int.jl")

@load "savedData/true.jld2" θt yt
@load "savedData/InferenceSamples.jld2" θi yi

# ==============================================
# ==========    Analytic Posterior    ==========
# ==============================================

AnalyticPosterior = Beta(1+sum(yt[2:end]), 1+sum(yt[1:end-1])-sum(yt[2:end]))
Ek = int(p -> -log(p)*pdf(AnalyticPosterior, p), 0, 1)
Vk = int(p -> (-log(p)-Ek)^2*pdf(AnalyticPosterior, p), 0, 1)
p = fplot((0.1, 0.5), k -> exp(-k)*pdf(AnalyticPosterior, exp(-k)), label = "Posterior Density")
vline!(p, [θt], label = "True Value")
scatter!(p, [Ek], [0.5*ylims(p)[2]], xerr = 2*sqrt(Vk), lw = 2, label = L"\mathbb{E}[\hat{k}] \pm 2\sqrt{\mathbb{V}[\hat{k}]}")

int(k -> exp(-k) * pdf(AnalyticPosterior, exp(-k)), 0, Inf)

# ==============================================
# ============    ABC Posterior    =============
# ==============================================

ϵ_ABC = mapslices(x -> norm(yt - x), yi, dims = 1)[:]
idx = sortperm(ϵ_ABC)
n = 1000
ABCPosterior = θi[idx[1:n]]
Ek_ABC = mean(ABCPosterior)
Vk_ABC = var(ABCPosterior)/n
KDE_ABC = kde(ABCPosterior)
p = fplot((0.1, 0.5), k -> pdf(KDE_ABC, k), label = "ABC Posterior Density")
vline!(p, [θt], label = "True Value")
scatter!(p, [Ek_ABC], [0.5*ylims(p)[2]], xerr = 2*sqrt(Vk_ABC), lw = 2, label = L"\mathbb{E}[\hat{k}] \pm 2\sqrt{\mathbb{V}[\hat{k}]}")

# ==============================================
# ====    Sufficient Statistic Posterior    ====
# ==============================================

t_SS =  (sum(yt[1:end-1])-sum(yt[2:end])) / sum(yt[2:end])
SS = mapslices(y -> (sum(y[1:end-1])-sum(y[2:end])) / sum(y[2:end]), yi, dims = 1)[:]

ϵ_SS = map(x -> norm(t_SS - x), SS)
idx = sortperm(ϵ_SS)
n = 1000
SSPosterior = θi[idx[1:n]]
Ek_SS = mean(SSPosterior)
Vk_SS = var(SSPosterior)/n
KDE_SS = kde(SSPosterior)
p = fplot((0.1, 0.5), k -> pdf(KDE_SS, k), label = "Sufficient Statistic Posterior Density")
vline!(p, [θt], label = "True Value")
scatter!(p, [Ek_ABC], [0.5*ylims(p)[2]], xerr = 2*sqrt(Vk_ABC), lw = 2, label = L"\mathbb{E}[\hat{k}] \pm 2\sqrt{\mathbb{V}[\hat{k}]}")

# ==============================================
# ============    PCA Posterior    =============
# ==============================================

@load "savedData/PCATransform.jld2" PCA_Transform

st = PCA_Transform' * yt[2:end]
si = (yi[2:end,:]' * PCA_Transform)'
ϵ_PCA = mapslices(x -> norm(st - x), si, dims = 1)[:]
idx = sortperm(ϵ_PCA)
n = 1000
PCAPosterior = θi[idx[1:n]]
Ek_PCA = mean(PCAPosterior)
Vk_PCA = var(PCAPosterior)/n
KDE_PCA = kde(PCAPosterior)
p = fplot((0.1, 0.5), k -> pdf(KDE_PCA, k), label = "PCA Posterior Density")
vline!(p, [θt], label = "True Value")
scatter!(p, [Ek_PCA], [0.5*ylims(p)[2]], xerr = 2*sqrt(Vk_PCA), lw = 2, label = L"\mathbb{E}[\hat{k}] \pm 2\sqrt{\mathbb{V}[\hat{k}]}")

# ==============================================
# ========    Autoencoder Posterior    =========
# ==============================================

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
idx = sortperm(ϵ_AE)
n = 1000
AEPosterior = θi[idx[1:n]]
Ek_AE = mean(AEPosterior)
Vk_AE = var(AEPosterior)/n
KDE_AE = kde(AEPosterior)
p = fplot((0.1, 0.5), k -> pdf(KDE_AE, k), label = "AE Posterior Density")
vline!(p, [θt], label = "True Value")
scatter!(p, [Ek_AE], [0.5*ylims(p)[2]], xerr = 2*sqrt(Vk_AE), lw = 2, label = L"\mathbb{E}[\hat{k}] \pm 2\sqrt{\mathbb{V}[\hat{k}]}")


# ==============================================
# ==========    Density Comparison    ==========
# ==============================================

p = plot(xlabel = "k", ylabel = "Posterior Density")
fplot!(p, (0.1, 0.5), k -> exp(-k)*pdf(AnalyticPosterior, exp(-k)), label = "True Posterior Density")
fplot!(p, (0.1, 0.5), k -> pdf(KDE_ABC, k), label = "ABC Posterior Density")
fplot!(p, (0.1, 0.5), k -> pdf(KDE_SS, k), label = "SS Posterior Density")
fplot!(p, (0.1, 0.5), k -> pdf(KDE_PCA, k), label = "PCA Posterior Density")
fplot!(p, (0.1, 0.5), k -> pdf(KDE_AE, k), label = "AE Posterior Density")

savefig(p, "../Assignment2Writing/figures/KDEs.pdf")