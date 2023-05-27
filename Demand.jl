# Import required libraries
using Parameters, LinearAlgebra, SparseArrays, Statistics, DataFrames, FixedEffectModels, Vcov, Setfield
include("Utilities.jl")

# Structure for results
struct Demand
    # Results
    val::Float64         # Value of objective
    θ₁::Array{Float64}   # Linear coefficients
    s₁::Array{Float64}   # Standard error on linear coefficients
    θ₂::Array{Float64}   # Non-linear coefficients
    # s₂::Array{Float64}   # Standard error on non-linear coefficients
    ξ::Array{Float64}    # Residual terms from GMM objective
    # sᵢ::Array{Float64}   # Individual choice probabilities
    # Δ::Array{Float64}    # Jacobian matrix
    # ε::Array{Float64}    # Elasticities
end

# Contraction mapping within a market
function ContractionMappingMarket(δ₀, μₘ, shares, R::Int64; tol::Float64=1e-10, maxiter::Int64=1000, verbose::Bool=false)
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
        if verbose
            println("Iteration i = ", i, " with error ε = ", error)
        end

        # Set new baseline
        δ₀ = δ
    end

    # Return values
    return δ
end

# Contraction mapping
function ContractionMapping(δ₀, X₂, shares, ν, θ₂, markets; tol::Float64=1e-12, maxiter::Int64=1000, verbose::Bool=false) #, D=nothing, Π=nothing
    # Initialize
    δ = zeros(size(markets, 1))

    # Iterate over unique markets
    for (m, market) in enumerate(unique(markets))
        # Filter to market and contract
        index = (markets .== market)

        # Compute δ by market
        R = size(ν[m], 1)
        μₘ = μ(X₂[index,:], ν[m], θ₂, R) # might need to change ν[m] index for matrix; need ν to be M x R x nl_var
        δ[index] = ContractionMappingMarket(δ₀[index], μₘ, shares[index], R, tol=tol, maxiter=maxiter, verbose=verbose)
    end

    # Return value
    return δ
end

# Objective function
function GMM(θ₂, data, ins_vars, ex_vars, nl_vars, markets, Z, W, ν, row_index, column_index; tol::Float64=1e-10, maxiter::Int64=1000, verbose::Bool=false)
    # Re-construct matrix
    θ₂ = Array(sparse(row_index, column_index, θ₂))

    # Contraction mapping
    data.delta = ContractionMapping(data.delta_iia, data[!, nl_vars], data.share, ν, θ₂, markets, tol=tol, maxiter=maxiter, verbose=verbose)

    # Fixed effects regression
    model = FixedEffectModels.reg(data, Term.(:delta) ~ (Term.(:price) ~ sum(Term.(Symbol.(ins_vars)))) + sum(Term.(Symbol.(ex_vars))) + fe(:Year), Vcov.robust(), save = true)
    θ₁ = model.coef
    s₁ = sqrt.(diag(model.vcov))

    # Obtain residuals and compute GMM criterion
    ξ = FixedEffectModels.residuals(model)
    g = mean(Z .* repeat(ξ, outer=[1, size(Z, 2)]), dims=1)
    val = g * inv(W) * g'

    # Return value
    return val[1], θ₁, s₁, ξ
end

# Standard errors
function BLPSE()
    return nothing
end

# Full model
function BLP(data::DataFrame, θ::Array{Float64}, ins_vars, ex_vars, nl_vars, markets, ν; tol::Float64=1e-12, maxiter::Int64=1000, verbose::Bool=false)
    # Print statement
    println("Beginning to fit BLP model...")

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
    results = Demand(0.0, [0.0], [0.0], θ, [0.0])
    Z = Matrix(data[!, ins_vars])
    W = Z' * Z # TODO: scale by 1 / J (num products)

    # Objective function wrapper
    function obj(θ₂)
        # Call original GMM criterion
        val, θ₁, s₁, ξ = GMM(θ₂, data, ins_vars, ex_vars, nl_vars, markets, Z, W, ν, row_index, column_index; tol=tol, maxiter=maxiter, verbose=verbose)
        
        # Set values in results
        results = @set results.val = val
        results = @set results.θ₁ = θ₁
        results = @set results.s₁ = s₁
        results = @set results.ξ = ξ

        # Return value
        return val
    end

    # Print statement
    println("First stage...")

    # Optimize first stage GMM
    S₁ = optimize(θ₂ -> obj(θ₂), θ₂, LBFGS()) # TODO: allow bounds input, optim method
    θ₂ = S₁.minimizer

    # Print statement
    println("Second stage...")

    # Update weight matrix and re-optimize
    W = (Z .* results.ξ)' * (Z .* results.ξ) # TODO: scale by 1 / J (num products)
    S₂ = optimize(θ₂ -> obj(θ₂), S₁.minimizer, LBFGS()) # TODO: allow bounds input, optim method
    θ₂ = Array(sparse(row_index, column_index, S₂.minimizer))
    results = @set results.θ₂ = θ₂

    # Print statement
    println("Finished!")

    # Return results
    return results
end
