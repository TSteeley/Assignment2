function int(f::Function, a::Number, b::Number; h::Number = 25)
    if !isinf(a) && !isinf(b)
        g = x -> a + 0.5(tanh(π/2 * sinh(x/h))+1) * (b-a)
        dg = x -> (b-a)/4*cosh(x/h)*(sech(π/2*sinh(x/h)))^2
    elseif isinf(a) && isinf(b)
        g = x -> π * sinh(x/h)
        dg = x -> cosh(x/h)
    elseif isinf(a)
        g = x -> log(0.5*tanh(π/2*sinh(x/h))+0.5)+b
        dg = x -> cosh(x/h)/(1+exp(π*sinh(x/h)))
    elseif isinf(b)
        g = x -> a-log(0.5*tanh(π/2*sinh(x/h))+0.5)
        dg = x -> cosh(x/h)/(1+exp(π*sinh(x/h)))
    end
    F = x -> (f∘g)(x)
    return (dg(0)*F(0) + sum(i -> dg(i)*F(i)+dg(-i)*F(-i), 1:ceil(Int, 3.1h)))*π/h
end