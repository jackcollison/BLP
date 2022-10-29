# Import required libraries
using Parameters, LinearAlgebra

# Multinomial logit function
function p(x)
    # Overflow trick
    m = max(0.0, maximum(x))
    n = exp.(x .- m)

    # Return value
    return n ./ (exp.(-m) + sum(n))
end

# Choice probability
function Λ(δ::Array{Float64}, X::Array{Float64}, ν::Array{Float64}, Σ::Array{Float64}, R::Int64; D::Array{Float64}=nothing, Π::Array{Float64}=nothing)
    # Initialize BxR values
    s = zeros(size(δ, 1), R)

    # Iterate over individuals
    for i = 1:R
        # Increment value
        x = δ + (ν[i,:] .* X) * Σ

        # Check for demographics
        if Π != nothing && Π != nothing
            # Add demographics
            x += sum((D[i, :] .* X) * Π, dims=2)
        end

        # Evaluate probability
        s[:, i] = p(x)
    end

    # Return value
    return s
end

# Simulated shares
function σ(δ::Array{Float64}, X::Array{Float64}, ν::Array{Float64}, Σ::Array{Float64}, R::Int64; D::Array{Float64}=nothing, Π::Array{Float64}=nothing)
    # Return brand-wise mean of individual market shares
    mean(Λ(δ, X, ν, Σ, R, D=D, Π=Π), dims=1)
end

# Contraction mapping within a market
function contraction_mapping_market(x)
    return nothing
end

# Contraction mapping
function contraction_mapping(x)
    return nothing
end

# Objective function
function gmm(x)
    return nothing
end

# Jacobian
function jacobian(x)
    return nothing
end

# Standard errors
function se(x)
    return nothing
end

# Full model
function blp(x)
    return nothing
end
