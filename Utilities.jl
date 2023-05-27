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

# Elasticities
function ε()
    return nothing
end

# Individual choice probabilities
function Λ()
    return nothing
end