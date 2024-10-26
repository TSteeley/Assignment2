@userplot squarify

@recipe function f(h::squarify)
    y, x = h.args

    Y = repeat(y, inner = 2)[2:end]
    X = repeat(x,inner = (2,1))[1:end-1,:]

    @series begin
        Y,X
    end
end


############
### Test ###
############

# # Case 1: t::Vector{Number} independent variable, correspends to the dependent rows of A::Matrix{Number}.
# t, A = alg1(50, 0.01, 30, 20, 0.1)

# squarify(t,A[:,1], label = "")

# # # Case 2: t::Vector{Vector{Number}} is an independent variable, corresponding to A::Vector{Vector{Number}}
# t, A = alg2(50, 30, 20, 0.1)

# squarify.(t,A) # Returns Vector{Plots.plot}

# p = plot()
# squarify(t[1],A[1])

# map(i -> squarify!(p, t[i], A[i]), 1:length(t))