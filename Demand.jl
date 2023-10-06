# Import required libraries
using Parameters, LinearAlgebra, SparseArrays, Statistics, DataFrames, FixedEffectModels, Vcov, Optim, Setfield
include("Utilities.jl")

# Structure for results
struct Demand
    # Results
    val::Float64         # Value of objective
    θ₁::Array{Float64}   # Linear coefficients
    s₁::Array{Float64}   # Standard error on linear coefficients
    θ₂::Array{Float64}   # Non-linear coefficients
    s₂::Array{Float64}   # Standard error on non-linear coefficients
    δ::Array{Float64}    # Mean utilities
    ξ::Array{Float64}    # Residual terms from GMM objective
end

# Contraction mapping within a market
function ContractionMappingMarket(δ₀, μₘ, shares, R, tol, maxiter, verbose)
    # Initialize values
    error = Inf
    δ = nothing
    i = 0

    # Iterate while error is large
    while error > tol && i <= maxiter
        # Increment counter
        i += 1

        # Compute choice probabilities
        σᵢ = P(δ₀ .+ μₘ)
        σₘ = sum(σᵢ, dims=2) / R

        # Update with contraction mapping
        δ = δ₀ + log.(shares) - log.(σₘ)
        error = maximum(abs.((δ - δ₀)))

        # Print statement
        if verbose > 1
            println("Iteration i = ", i, " with error ε = ", error)
        end

        # Set new baseline
        δ₀ = δ
    end

    # Return values
    return δ
end

# Contraction mapping
function ContractionMapping(δ₀, X₂, shares, ν, θ₂, markets, tol, maxiter, verbose)
    # Initialize
    δ = zeros(size(markets, 1))

    # Iterate over unique markets
    for (m, market) in enumerate(unique(markets))
        # Filter to market and contract
        index = (markets .== market)

        # Compute δ by market
        R = size(ν[m, :, :], 1)
        μₘ = μ(X₂[index,:], ν[m, :, :], θ₂, R)
        δ[index] = ContractionMappingMarket(δ₀[index], μₘ, shares[index], R, tol, maxiter, verbose)
    end

    # Return value
    return δ
end

# Objective function
function GMM(θ₂, data, ins_vars, ex_vars, nl_vars, markets, Z, W, ν, row_index, column_index, τ, ds, tol, maxiter, verbose)
    # Re-construct matrix
    θ₂ = Array(sparse(row_index, column_index, θ₂))

    # Contraction mapping
    data.delta = ContractionMapping(data.delta_iia, data[!, nl_vars], data.share, ν, θ₂, markets, tol, maxiter, verbose)

    # Fixed effects regression
    model = FixedEffectModels.reg(data, Term.(:delta) ~ (Term.(:price) ~ sum(Term.(Symbol.(ins_vars)))) + sum(Term.(Symbol.(ex_vars))) + fe(:Year), Vcov.robust(), save=true)
    θ₁ = model.coef
    s₁ = sqrt.(diag(model.vcov))

    # Obtain residuals
    δ = data.delta
    ξ = FixedEffectModels.residuals(model)
    g₀ = mean(Z .* repeat(ξ, outer=[1, size(Z, 2)]), dims=1)

    # Extra moments
    G₁ = 0
    if τ !== nothing && ds !== nothing
        dŝ = dsdτ(results, data.type, data[!, nl_vars], markets, ν, τ)
        ê = dŝ - ds
        G₁ = ê * inv(W) * ê'
    end

    # Return GMM criterion
    val = g₀ * inv(W) * g₀' .+ G₁
    return val[1], θ₁, s₁, δ, ξ
end

# Standard errors
function StandardErrors(results, data, nl_vars, markets, Z, W, ν)
    # Pre-compute values
    X₂ = Matrix(data[!, nl_vars])
    M = dδdθ₂(results, X₂, markets, ν)
    J = size(X₂, 1)

    # Intermediate values
    Γ = (1 / J) * (Z' - Z' * X₂ * inv(X₂' * Z * inv(W) * Z' * X₂) * X₂' * Z * inv(W) * Z') * M
    V = Γ' * inv(W) * Γ
    
    # Return values
    return sqrt.(diag(V) ./ J)
end

# Full model
function BLP(data::DataFrame, θ::Array{Float64}, ins_vars, ex_vars, nl_vars, ν; τ=nothing, ds=nothing, tol::Float64=1e-12, maxiter::Int64=1000, verbose::Int64=0)
    # Print statement
    if verbose > 0
        println("Beginning to fit BLP model...")
    end

    # Get indices
    index = findall(x -> x != 0, θ)
    row_index = getindex.(index, [1])
    column_index = repeat([1], size(row_index,1))
    
    # Check dimensions
    if typeof(index) == Vector{CartesianIndex{2}}
        # Update index
        column_index = getindex.(index, [2])
    end

    # Flatten parameters
    θ₂ = θ[index]

    # Initialize results
    J = size(data, 1)
    results = Demand(0.0, [0.0], [0.0], θ, [0.0], [0.0], [0.0])
    markets = data.market_id
    Z = Matrix(data[!, ins_vars])
    W = (1 / J) * Z' * Z

    # Objective function wrapper
    function obj(θ₂)
        # Call original GMM criterion
        val, θ₁, s₁, δ, ξ = GMM(θ₂, data, ins_vars, ex_vars, nl_vars, markets, Z, W, ν, row_index, column_index, τ, ds, tol, maxiter, verbose)
        
        # Set values in results
        results = @set results.val = val
        results = @set results.θ₁ = θ₁
        results = @set results.s₁ = s₁
        results = @set results.δ = δ
        results = @set results.ξ = ξ

        # Return value
        return val
    end

    # Print statement
    if verbose > 0
        println("First stage...")
    end

    # Optimize first stage GMM
    S₁ = optimize(θ₂ -> obj(θ₂), θ₂, LBFGS())
    θ₂ = S₁.minimizer

    # Print statement
    if verbose > 0
        println("Second stage...")
    end

    # Update weight matrix and re-optimize
    W = (1 / J) * (Z .* results.ξ)' * (Z .* results.ξ)
    S₂ = optimize(θ₂ -> obj(θ₂), S₁.minimizer, LBFGS())
    θ₂ = Array(sparse(row_index, column_index, S₂.minimizer))
    results = @set results.θ₂ = θ₂

    # Compute standard errors
    s₂ = StandardErrors(results, data, nl_vars, markets, Z, W, ν)
    results = @set results.s₂ = Array(sparse(row_index, column_index, s₂))

    # TODO: allow bounds, optimization method choice

    # Print statement
    if verbose > 0
        println("Finished!")
    end

    # Return results
    return results
end
