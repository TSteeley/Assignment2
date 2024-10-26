@userplot int_plot

@recipe function f(plt::int_plot; h = 10, lb = 0)
    X, fun = plt.args
    if length(X) == 2
        X = range(X..., length = 250)
    end
    Y = integrate.(fun, lb, X, h = h)
    @series begin
        X,Y
    end
end

function integrate(f::Function, a::Number, b::Number; h::Number = 25)
    g = x -> a + 0.5(tanh(π/2 * sinh(x))+1) * (b-a)
    dg = x -> π*(b-a)/(4h)*cosh(x)*(sech(π/2*sinh(x)))^2
    F = x -> (f∘g)(x)
    return dg(0)*F(0) + sum(map(i -> dg(i/h)*(F(i/h)+F(-i/h)), 1:ceil(Int, 2.5h)))
end
