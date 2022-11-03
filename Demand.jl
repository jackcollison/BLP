# Import required libraries
using Parameters, LinearAlgebra, SparseArrays
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
            x += sum((D[i, :] .* X) * Π)
            # TODO: Check dims=2
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
    return mean(Λ(δ, X, ν, Σ, R, D=D, Π=Π), dims=2)
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
        δ = (δ₀ .* shares) ./ σ(δ₀, X, ν, Σ, R, D=D, Π=Π)
        # TODO: If close enough, we can take a Newton step
    end

    # Return value
    return log.(δ)
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

        # Check demographics
        Dₘ = nothing
        if D !== nothing
            Dₘ = D[m,:,:]
        end

        # Compute δ by market
        δ[index] = contraction_mapping_market(δ[index], δ₀[index], X[index,:], shares[index], ν[m,:,:], Σ, R, tol=tol, maxiter=maxiter, D=Dₘ, Π=Π)
    end

    # Return value
    return δ
end

# Objective function
function gmm(θ₂::Array{Float64}, X₁::Array{Float64}, X₂::Array{Float64}, Z::Array{Float64}, shares::Array{Float64}, δ₀::Array{Float64}, markets::Array{Float64}, W::Array{Float64}, ν::Array{Float64}; D::Array{Float64}=nothing, index::Array{Float64}=nothing, verbose::Bool=false, tol::Float64=1e-10)
    # Note: Index is a list of two with row and column indices of zeros
    # TODO: Test indexing works

    # Initialize Σ and Π
    Σ = nothing
    Π = nothing
    # TODO: Convert to sparse array version

    # Contraction mapping
    δ = contraction_mapping(δ₀, X₂, shares, ν, Σ, markets, tol=tol, maxiter=maxiter, verbose=verbose, D=D, Π=Π)

    # Fit linear model
    m = fit(X₁, δ, Z, "IVGMM")
    θ₁ = m.θ
    s₁ = m.s

    # Obtain residuals and compute GMM criterion
    ξ = δ - X₁ * θ₁
    g = mean(Z' * ξ, dims=2)
    val = g' * W * g

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

    # Get indices of non-zero elements
    Σᵢ = findall(x -> x != 0, Σ)
    Πᵢ = nothing 
    if D !== nothing && Π !== nothing 
        Πᵢ = findall(x -> x != 0, Π)
    end
    index = [Σᵢ, Πᵢ]

    # Flatten for optimization routine
    θ₂ = Σ[Σᵢ]
    if Π !== nothing
        Πₓ = Π[Πᵢ]
        θ₂ = vcat(θ₂, Πₓ[:])
    end

    # Objective function wrapper
    function obj(θ₂)
        # Call original GMM criterion
        val, results.θ₁, results.s₁, ξ = gmm(θ₂, X₁, X₂, Z, shares, δ₀, markets, W, ν, D=D, index=index, verbose=verbose, tol=tol)
        return val
    end

    # Optimize first stage GMM
    # TODO: Implement
end

# Structure for results
struct Demand
    # Results
    θ₁::Array{Float64}
    s₁::Array{Float64}

    # TODO: Store p(x)
    # TODO: Store αᵢ
end