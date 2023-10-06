# Import required libraries
using LinearAlgebra

# Multinomial logit function
function P(x)
    # Overflow trick
    m = max(0.0, maximum(x))
    n = exp.(x .- m)

    # Return value
    return n ./ (exp.(-m) .+ sum(n, dims=1))
end

# Find μ
function μ(X, ν, θ₂, R)
    # Initialize results
    m = zeros(size(X, 1), R)
    J = size(X, 2)

    # Iterate over R, J
    for i = 1:R
        for j = 1:J
            # Increment value
            m[:,i] += (ν[i, j] * θ₂[:, j]) .* X[:, j]
        end
    end

    # Return value
    return m
end

# Price derivative
function PriceDerivative(α, σₚ, νₚ, sⱼ, sₖ, own::Bool)
    # Own- versus cross-price
    if own
        # Compute derivative
        return mean((α .+ σₚ * νₚ) .* sⱼ .* (1 .- sₖ))
    else
        # Compute derivative
        return -mean((α .+ σₚ * νₚ) .* sⱼ .* sₖ)
    end
end

# Elasticities for a given market
function εₘ(results, prices, shares, ν, σᵢ)
    # Extract coefficients
    α = results.θ₁[end]
    σₚ = results.θ₂[1]
    J = size(prices, 1)

    # Assume ν structure
    # TODO: Generalize for more terms
    νₚ = ν[:, 1]

    # Iterate over products
    elasticities = zeros(J, J)
    for j = 1:J
        for k = 1:J
            # Compute elasticitity
            dp = PriceDerivative(α, σₚ, νₚ, σᵢ[j, :], σᵢ[k, :], j == k)
            elasticities[j, k] = dp * prices[k] / shares[j]
        end
    end

    # Return values
    return elasticities
end

# Get elasticities
function ε(results, data, nl_vars)
    # Initialize
    X₂ = data[!, nl_vars]
    elasticities = Matrix{Float64}[]

    # Iterate over unique markets
    for (m, market) in enumerate(unique(markets))
        # Filter to market and contract
        index = (markets .== market)
        νₘ = ν[m]

        # Choice probabilities
        R = size(νₘ, 1)
        X₂ₘ = X₂[index, :]
        μₘ = μ(X₂ₘ, νₘ, results.θ₂, R)
        δₘ = results.δ[index]
        σᵢ = P(δₘ .+ μₘ)

        # Compute elasticities
        prices = X₂ₘ[:, 1]
        shares = data.share[index]
        e = εₘ(results, prices, shares, νₘ, σᵢ)

        # Concatenate
        push!(elasticities, e)
    end

    # Return object
    return elasticities
end

# Own elasticity
function OwnElasticities(results, data, nl_vars; elasticities=nothing)
    # Check if already computed
    if elasticities === nothing
        # Compute elasticities
        elasticities = ε(results, data, nl_vars)
    end

    # Own-price elasticity
    N = length(elasticities)
    e = []
    for i in 1:N
        # Append own-price elasticities
        append!(e, diag(elasticities[i]))
    end

    # Return values
    return e
end

# Gradient object
function dδdθ₂(results, X₂, markets, ν)
    # Initialize
    M = []

    # Iterate over unique markets
    for (m, market) in enumerate(unique(markets))
        # Filter to market and contract
        index = (markets .== market)
        νₘ = ν[m]

        # Choice probabilities
        R = size(νₘ, 1)
        X₂ₘ = X₂[index, :]
        μₘ = μ(X₂ₘ, νₘ, results.θ₂, R)
        δₘ = results.δ[index]
        σᵢ = P(δₘ .+ μₘ)

        # Generate dsdδ
        Jₘ = size(X₂ₘ, 1)
        dsdδ = -(1 / R) * σᵢ * σᵢ'
        dsdδ[diagind(dsdδ)] = (1 / R) * diag(σᵢ * (1 .- σᵢ)')

        # Generate dsdθ₂
        dsdθ₂ = zeros(Jₘ, size(X₂, 2))
        for k = 1:size(X₂, 2)
            for i = 1:R
                # Increment value
                dsdθ₂[:, k] += (1 / R) * (νₘ[i, k] * σᵢ[:, i] .* (X₂ₘ[:, k] .- X₂ₘ[:, k]' * σᵢ[:, i]))
            end
        end

        # Concatenate
        M = vcat(M, -inv(dsdδ) * dsdθ₂)
    end

    # Return object
    return M
end

# Tax incidence moments
function dsdτ(results, products, X₂, markets, ν, τ)
    # Initialize
    α = results.θ₁[end]
    σₚ = results.θ₂[1]
    M = []

    # Iterate over unique markets
    for (m, market) in enumerate(unique(markets))
        # Filter to market and contract
        index = (markets .== market)
        νₘ = ν[m]
        νₚ = νₘ[:, 1]
        αᵢ = α .+ σₚ .* νₚ

        # Inclusive values
        R = size(νₘ, 1)
        X₂ₘ = X₂[index, :]
        μₘ = μ(X₂ₘ, νₘ, results.θ₂, R)
        δₘ = results.δ[index]
        Vᵢ = exp.(δₘ .+ μₘ)

        # Initialize
        J = sum(index)
        mo = zeros(J)

        # Construct values
        dpdτΔτ = zeros(J)
        for (j, pₘ) in enumerate(products[index])
            # Initialize
            dpdτΔτ[j] = 0.0

            # Update value
            if haskey(τ, pₘ)
                dpdτΔτ[j] = τ[pₘ]
            end
        end

        # Iterate over products
        for j in 1:J
            # Compute moments
            D = 1 .+ sum(Vᵢ[1:end .!= j, :], dims=1)'
            mo[j] = sum(Vᵢ[j, :] .* αᵢ .* (dpdτΔτ[j] .* D .- sum(dpdτΔτ[1:end .!= j] .* Vᵢ[1:end .!= j, :])) ./ D.^2) / R
        end

        # Concatenate
        M = vcat(M, mo)
    end

    # Return object
    return M
end