
# Custom Layer to split data, this causes the indexing warning
struct SplitData{T}
    positions::T
end

SplitData(positions...) = SplitData([0 ; positions...])
SplitData(p::Int) = SplitData([0 ; p])

Flux.@functor SplitData

# Avoid scalar indexing by using view instead of direct indexing
function (m::SplitData)(x::AbstractArray)
    map(p -> view(x, p[1]+1:p[2], :), zip(m.positions[1:end], [m.positions[2:end]; size(x, 1)]))
end

# INCA Network

# Encoder layer
function Encoder(q::Int)
    return Chain(
        Conv((3,), 1  => 16, relu; pad = 0),
        Conv((3,), 16 => 16, relu; pad = 0),
        MaxPool((2,)),
        Conv((3,), 16 => 32, relu; pad = 0),
        Conv((3,), 32 => 32, relu; pad = 0),
        Conv((3,), 32 => q; pad = 0),
        GlobalMeanPool(),
    )
end

function ReshapeINCA(q::Int)
    return Chain(
        Flux.flatten,
        SplitData(q)
    )
end

function decoderINCA(q::Int, p::Int)
    return Chain(
        Dense(q-p => 3, x -> leakyrelu(x,0.3)),
        Dense(3 => 10,  x -> leakyrelu(x,0.3)),
        Dense(10 => p,  x -> leakyrelu(x,0.3)),
        # Dense(3 => 1, sigmoid)
    )
end

struct INCA 
    encoder
    reshape
    decoder
end

Flux.@functor INCA

function INCA(q::Int, p::Int)
    m1 = Encoder(q)
    m2 = ReshapeINCA(p)
    m3 = decoderINCA(q, p)
    return INCA(m1, m2, m3)
end

function (m::INCA)(X)
    s = m.encoder(reshape(X, size(X,1), 1, :))
    s,s2 = m.reshape(s)
    θhat = m.decoder(s2)
    return s, θhat
end