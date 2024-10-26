@userplot corr_plot

"""
    args:
        θt::Matrix 

    kwargs:
        name::Vector{String} Variable names
        true_vals::Vector{Real} True Values

"""
@recipe function f(h::corr_plot; name = nothing, true_vals = [])
    X = h.args[1]
    # disc = h.args[2]
    M = size(X,1)

    if name === nothing
        name = [LaTeXString("\$k_$(i)\$") for i in 1:M]
    end

    titles = Matrix{Any}(undef,M,M)
    titles .= ""
    for (idx, name) in enumerate(name)
        titles[idx,1] = name
    end
    titles = reshape(titles, 1, M^2)

    w = get(plotattributes, :weights, ones(size(X,2)))

    layout := @layout [
        grid(M,M)
    ]
    margin := 0.3Plots.cm
    size := (700, 700)
    legend := false
    xrot := 35


    for i=1:M, j=1:M # i columns, j rows
        Yname = j == 1 ? name[i] : ""
        if i == j
            @series begin
                seriestype := :histogram
                normalize := :pdf
                bins := 25
                ylabel := Yname
                subplot := (i-1)*M+j
                title := titles[(i-1)*M+j]
                
                X[i,:]
            end
            if ! isempty(true_vals)
                @series begin
                    seriestype := :vline
                    lw := 3
                    subplot := (i-1)*M+j
                    title := titles[(i-1)*M+j]
                    ylabel := Yname

                    [true_vals[i]]
                end
            end
            # println(get(plotattributes, :ylim, :auto))
            # @series begin
            #     xerr := StatsBase.std(X[i,:], StatsBase.Weights(w))
            #     subplot := (i-1)*M+j

            #     [StatsBase.mean(X[i,:], StatsBase.Weights(w))], [0.5*(get(plotattributes, :ylim, :auto)[2])]
            # end
            # p1 = histogram(X[i,:], normalize = :pdf, bins = 25, ylabel = Yname)
            # push!(plots, p1)
        elseif i < j
            k = kde((X[j,:], X[i,:]), weights = w)
            @series begin
                seriestype := :contour
                levels := 5
                fill := false
                colorbar := false
                subplot := (i-1)*M+j
                title := titles[(i-1)*M+j]
                # xlims := (xmin, xmax)
                # ylims := (ymin, ymax)
                # color := cgrad(C(colours[i], 3levels))

                (collect(k.x), collect(k.y), k.density')
            end
            # push!(plots, contourf(kde((X[i,:], X[j,:])), c = :turbo, cbar = false, lw = 0.0, lc = "black"))
        else
            @series begin
                seriestype := scatter
                ylabel := Yname
                subplot := (i-1)*M+j
                title := titles[(i-1)*M+j]

                X[j,:], X[i,:]
            end
            # push!(plots, scatter(X[i,:], X[j,:], ylabel = Yname))
        end
    end
end

@userplot acf
@recipe function acf(h::acf; Range = 1:30)
    n = length(Range)
    y = autoCorr(h.args[1], Range)
    y = vec([zeros(n) y zeros(n)]')
    x = repeat(Range, inner = 3)

    lc := "black"
    lw := 3
    
    @series begin
        x,y
    end
    # plot!(p[i], Range,  unc, lw = 2, color = "blue", label = "95% Confidence Interval", ls = :dash)
    # plot!(p[i], Range, -unc, lw = 2, color = "blue", label = "", ls = :dash)
end

function autoCorr(x::Vector, R::UnitRange{Int64})
    x = x .- mean(x)
    n = length(x)
    V = [x[1:end-i] ⋅ x[1+i:end] * (n-i)^-1 for i in R] ./ var(x)
    # unc = 1.96 ./ sqrt.([length(i) for i in data])
    return V
end

function est_MAP_CI(P::Vector)
    k = kde(P)
    midpoints, dens = k.x, k.density
    MAP = midpoints[findmax(dens)[2]]
    CI = cumsum(dens) / sum(dens) |> x -> midpoints[[findlast(x .< 0.025), findfirst(x .> 0.975)]]
    return MAP, CI
end