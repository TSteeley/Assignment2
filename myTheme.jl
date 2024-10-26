myTheme = Dict(
    :titlefontsize     => 12,
    :minorgridalpha    => 0.06,
    :minorgrid         => true,
    :framestyle        => :box,
    :legend            => :topright,
    :gridalpha         => 0.4,
    :background        => :white,
    :markerstrokewidth => 0,
    :colorgradient     => :magma,
    :tickfontsize      => 8,
    :guidefontsize     => 12,
    :gridlinewidth     => 0.7,
    :grid              => true,
    :linewidth         => 2.4,
    :minorticks        => 5,
    :fontfamily        => "Computer Modern",
)

PlotThemes._themes[:myTheme] = PlotTheme(myTheme)
theme(:myTheme)