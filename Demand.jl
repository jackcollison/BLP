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
function contraction_mapping_market(δ::Array{Float64}, δ₀::Array{Float64}, X::Array{Float64}, log_share::Array{Float64}, ν::Array{Float64}, Σ::Array{Float64}, R::Float64; tol::Float64=1e-10, verbose::Bool=false, D::Array{Float64}=nothing, Π::Array{Float64}=nothing)
    return nothing
end

# Contraction mapping
function contraction_mapping(δ₀::Array{Float64}, X::Array{Float64}, shares::Array{Float64}, ν::Array{Float64}, Σ::Array{Float64}, markets::Array{Float64}; tol::Float64=1e-10, verbose::Bool=false, D::Array{Float64}=nothing, Π::Array{Float64}=nothing)
    return nothing
end

# Objective function
function gmm(θ₂::Array{Float64}, W::Array{Float64}, ν::Array{Float64}; D::Array{Float64}=nothing, index::Array{Float64}=nothing)
    # TODO: Reshape θ₂ into Σ and Π
    # TODO: Ensure only values at index are entered
    # Note: Index is 4x5 and θ₂ is Kx1 where K is the number of non-zero elements of Σ and Π
    return nothing
end

# Jacobian
function jacobian(x)
    return nothing
end

# Standard errors
function blp_se(x)
    return nothing
end

# Full model
#X1, X2, Z, share, markets, ν, D, R, Σ, Π, δ_0, method='L-BFGS-B', tol_1=1e-5, tol_2=1e-10, verbose=False
function blp(x)
    return nothing
end

# Structure for results
struct Demand
end