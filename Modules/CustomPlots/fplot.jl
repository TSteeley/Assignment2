# Easy plot function
@userplot fplot
@recipe function f(h::fplot)
    X, fun = h.args

    if length(X) == 2
        X = range(X..., length = 250)
    end

    Y = fun.(X)
    @series begin
        X,Y
    end
end