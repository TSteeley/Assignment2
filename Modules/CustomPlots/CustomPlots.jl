module CustomPlots
    using Plots, StatsPlots, KernelDensity, Pipe, Distributions, ColorSchemes, LaTeXStrings

    export corrPlot, squarify, fplot, my_kde_plot, int_plot, est_MAP_CI

    include("correlation_Plots.jl")
    include("squarify.jl")
    include("marginalKDE.jl")
    include("plot_int.jl")
    include("fplot.jl")

end

