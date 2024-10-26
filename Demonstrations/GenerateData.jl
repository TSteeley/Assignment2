using Random, Distributions, Base.Threads, JLD2, ProgressBars

# Model
function decay(k::Number; N::Int=100, obs::Vector = [0:50;])
    t = [0 ; cumsum(-log.(rand(N)) ./ (k*[N:-1:1;]))] # Calculate time of each decay event
    return N .+ 1 .- [findlast(t .≤ o) for o in obs] # System state at times obs
end

prior = Exponential(1)

# ====================================================
# =========    Generate True Observation    ==========
# ====================================================

Random.seed!(0)
θt = 0.25
yt = decay(θt)
@save "savedData/true.jld2" θt yt

# ====================================================
# ==========    Prior Predictive Samples    ==========
# ====================================================

Random.seed!(1)
θp = rand(prior, 1_000_000)
yp = zeros(51, 1_000_000)
@threads for i in 1:1_000_000 |> ProgressBar
    @inbounds yp[:,i] = decay(θp[i])
end
@save "savedData/PriorPredictiveSamples.jld2" θp yp

# ====================================================
# ===========    Samples For Inference    ============
# ====================================================

Random.seed!(2)
θi = rand(prior, 10_000_000)
yi = zeros(51, 10_000_000)
@threads for i in 1:10_000_000 |> ProgressBar
    @inbounds yi[:,i] = decay(θi[i])
end
@save "savedData/InferenceSamples.jld2" θi yi