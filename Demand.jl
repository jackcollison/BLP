# Import required libraries
using Parameters, LinearAlgebra
include("Utilities.jl")

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
        if D !== nothing && Π !== nothing
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
    # Return predicted shares
    return mean(Λ(δ, X, ν, Σ, R, D=D, Π=Π), dims=1)
end

# Contraction mapping within a market
function contraction_mapping_market(δ::Array{Float64}, δ₀::Array{Float64}, X::Array{Float64}, shares::Array{Float64}, ν::Array{Float64}, Σ::Array{Float64}, R::Float64; tol::Float64=1e-10, maxiter::Int64=1000, D::Array{Float64}=nothing, Π::Array{Float64}=nothing)
    # Initialize counter
    i = 0

    # Iterate while error is large
    while maximum(abs.(δ .- δ₀)) > tol && i <= maxiter
        # Increment counter
        i += 1

        # Update previous and iterate
        δ₀ = δ
        δ += log.(shares) - σ(δ, X, ν, Σ, R, D=D, Π=Π)
    end

    # Return value
    return δ
end

# Contraction mapping
function contraction_mapping(δ₀::Array{Float64}, X::Array{Float64}, shares::Array{Float64}, ν::Array{Float64}, Σ::Array{Float64}, markets::Array{Float64}; tol::Float64=1e-10, maxiter::Int64=1000, verbose::Bool=false, D::Array{Float64}=nothing, Π::Array{Float64}=nothing)
    # Initialize δ
    δ = zeros(size(markets, 1))

    # Iterate over unique markets
    # TODO: Parallelize if it's more efficient
    for (m, market) in enumerate(unique(markets))
        # Filter to market and contract
        index = (markets .== market)
        δ[index] = contraction_mapping_market(δ[index], δ₀[index], X[index,:], shares[index], ν[m,:,:], Σ, R, tol=tol, maxiter=maxiter, D=D[m,:,:], Π=Π)
        # TODO: Handle case when D, Π are nothing objects because we can't apply the index mask
    end

    # Return value
    return δ
end

# Objective function
function gmm(θ₂::Array{Float64}, X₁::Array{Float64}, X₂::Array{Float64}, Z::Array{Float64}, shares::Array{Float64}, δ₀::Array{Float64}, markets::Array{Float64}, W::Array{Float64}, ν::Array{Float64}; D::Array{Float64}=nothing, index::Array{Float64}=nothing, verbose::Bool=false, tol::Float64=1e-10)
    # Note: Index is 4x5 binary matrix and θ₂ is of length K, where K is the number of non-zero elements of Σ and Π
    # TODO: Test indexing works

    # Initialize Σ and Π
    Σ = θ₂[1:size(X1, 2)]
    Π = nothing
    if size(θ₂) > size(X₁, 2)
        Π = zeros(size(index))
        counter = 1
        for i = 1:size(Π, 1)
            for j = 1:size(Π, 2)
                if index[i, j] > 0
                    Π[i, j] = θ₂[size(X1, 2) + counter]
                    counter += 1
                end
            end
        end
    end

    # Contraction mapping
    δ = contraction_mapping(δ₀, X₂, shares, ν, Σ, markets, tol=tol, maxiter=maxiter, verbose=verbose, D=D, Π=Π)

    # Fit linear model
    m = fit(X₁, δ, Z, "IVGMM")
    θ₁ = m.θ
    s₁ = m.s

    # Obtain residuals and compute GMM criterion
    ξ = δ - X₁ * θ₁
    g = mean(Z * ξ, dims=1)
    val = g.T * W * g

    # Return value
    return val, θ₁, s₁, ξ
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
function blp(results::Demand)
    # Print statement
    println("Beginning to fit BLP model...")

    # Objective function wrapper
    function obj(θ₂)
        # Call original GMM criterion
        val, results.θ₁, results.s₁, ξ = gmm(θ₂, X₁, X₂, Z, shares, δ₀, markets, W, ν, D=D, index=index, verbose=verbose, tol=tol)
    end
end

# Structure for results
struct Demand
    # Results
    θ₁::Array{Float64}
    s₁::Array{Float64}
end