function bounds(x::Vector, clip::Tuple)
    m_x = median(x)
    dx_l = m_x - quantile(x, 0.16)
    dx_h = quantile(x, 0.84) - m_x
    xmin = m_x + clip[1][1] * dx_l
    xmax = m_x + clip[1][2] * dx_h
    return xmin, xmax
end

function C(x::Vector{Int}, l)
    x1 = ones(Float64, 3)
    x /= 255.0
    return ColorScheme(
        [RGB((x1 .+ λ * (x - x1))...) for λ in LinRange(0.0,1.0,l)]
    )
end

@userplot my_kde_plot
@recipe function f(h::my_kde_plot)

    x, y = h.args

    layout := @layout [
        [density1 ; density2] contour{0.666w}
    ]
    legend --> false
    
    k = KernelDensity.kde((x, y))
    kx = KernelDensity.kde(x)
    ky = KernelDensity.kde(y)

    @series begin
        seriestype := :contour
        fill := false
        colorbar := false
        subplot := 3

        (collect(k.x), collect(k.y), k.density')
    end

    @series begin
        subplot := 1

        kx.x, kx.density
    end

    @series begin
        subplot := 2

        ky.x, ky.density
    end

end
# old
# @userplot my_kde_plot
# @recipe function f(h::my_kde_plot; levels = 10, trueVals = Dict(), colours=Plots.ColorGradient[])
#     clip = ((-3.0, 3.0), (-3.0, 3.0))

#     x1 = h.args[1][1][1,:]
#     y1 = h.args[1][1][2,:]
#     x2 = h.args[1][2][1,:]
#     y2 = h.args[1][2][2,:]

#     x1l, x1u = bounds(x1, clip)
#     y1l, y1u = bounds(y1, clip)
#     x2l, x2u = bounds(x2, clip)
#     y2l, y2u = bounds(y2, clip)

#     xmin = min(x1l, x2l)
#     xmax = max(x1u, x2u)

#     ymin = min(y1l, y2l)
#     ymax = max(y1u, y2u)

#     layout := @layout [
#         [density1 ; density2] contour{0.666w}
#     ]
#     legend --> false
    
#     for (i, (x,y)) in enumerate([(x1,y1), (x2,y2)])

#         k = KernelDensity.kde((x, y))
#         kx = KernelDensity.kde(x)
#         ky = KernelDensity.kde(y)

#         ps = pdf.(Ref(k), x, y)

#         ls = []
#         for p in range(1.0 / levels, stop = 1 - 1.0 / levels, length = levels - 1)
#             push!(ls, quantile(ps, p))
#         end

#         @series begin
#             seriestype := :contour
#             levels := ls
#             fill := false
#             colorbar := false
#             subplot := 3
#             xlims := (xmin, xmax)
#             ylims := (ymin, ymax)
#             color := cgrad(C(colours[i], 3levels))

#             (collect(k.x), collect(k.y), k.density')
#         end

#         # xguide := ""
#         # yguide := ""

#         @series begin
#             seriestype := :density
#             subplot := 1
#             xlims := (xmin, xmax)
#             ylims := (0, 1.1 * maximum(kx.density))
#             colour := RGB((colours[i] ./ 255)...)

#             x
#         end

#         @series begin
#             seriestype := :density
#             subplot := 2
#             ylims := (0, 1.1 * maximum(ky.density))
#             xlims := (ymin, ymax)
#             colour := RGB((colours[i] ./ 255)...)

#             y
#         end
#     end

#     if length(trueVals) == 2
#         K = keys(trueVals) |> collect
#         @series begin
#             seriestype := :vline
#             subplot := 1
#             xlabel := K[1]
#             lw := 2
#             lc := RGB(([226,111,70]./255)...)

#             [trueVals[K[1]]]
#         end

#         @series begin
#             seriestype := :vline
#             subplot := 2
#             xlabel := K[2]
#             lw := 2
#             lc := RGB(([226,111,70]./255)...)

#             [trueVals[K[2]]]
#         end

#         @series begin
#             seriestype := :hline
#             subplot := 3
#             lw := 2
#             lc := RGB(([226,111,70]./255)...)

#             [trueVals[K[2]]]
#         end

#         @series begin
#             seriestype := :vline
#             subplot := 3
#             lw := 2
#             lc := RGB(([226,111,70]./255)...)
#             xlabel := K[1]
#             ylabel := K[2]

#             [trueVals[K[1]]]
#         end
#     end
# end

# Test
# using Distributions, KernelDensity, ColorSchemes

# x = postPrior[1,:]
# y = postPrior[2,:]
# trueVals = Dict(
#     L"\mu_0" => 0.125,
#     L"\beta_P" => 1.0
# )
# p = myplot(
#     [postPrior, MpostPrior], 
#     trueVals = trueVals,
#     colours = ([123, 40, 255], [255, 19, 45])
# )

# c = contour(k.x,k.y,k.density', color = cgrad(C([123, 40, 255], 30)), lw = 2)


