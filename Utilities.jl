# Import required libraries
using LinearAlgebra

# Linear generalized method of moments
struct LGMM
    # Results
    type::String
    θ::Array{Float64}
    s::Array{Float64}
end

# Linear models
function estimate(X::Matrix{Float64}, Y::Matrix{Float64}, Z::Matrix{Float64}=nothing, W::Matrix{Float64}=nothing)
    # Check cases
    if Z == nothing && W == nothing
        # Return OLS estimate
        return inv(X.T * X) * X.T * Y
    elseif Z == nothing && W != nothing
        # Return FGLS estimate
        return inv(X.T * W * X) * X.T * W * Y
    elseif Z != nothing && W == nothing
        # Return IV estimate
        P = Z * inv(Z.T * Z) * Z.Τ
        return inv(X.T * P * X) * X.T * P * Y
    else
        # Return IV FGLS estimate
        return inv(X.T * Z * W * Z.T * X) * X.T * Z * W * Z.T * Y
    end
end

# Estimate weight matrix
function weight(X::Matrix{Float64}, Y::Matrix{Float64}, θ::Matrix{Float64}, Z::Matrix{Float64}=nothing)
    # Compute residual
    ε = Y - X * θ

    # Check cases
    if Z == nothing
        # Generalized least squares
        return diag(ε.^2)
    end

    # Instrumental variables
    return inv(Z.T * diag(ε.^2) * Z)
end

# Estimate standard errors
function se(X::Matrix{Float64}, Y::Matrix{Float64}, N::Float64, Z::Matrix{Float64}=nothing, W::Matrix{Float64}=nothing)
    # Initialize
    Γ = nothing

    # Check cases
    if Z == nothing
        # OLS standard error
        Γ = (1 / N) .* X * X
    else
        # IV standard error
        Γ = (1 / N) .* Z.T * X
    end

    # Return standard errors
    if W == nothing
        # No weighting matrix
        return sqrt.((1 / N) .* diag(inv(Γ.T * Γ)))
    else
        # Use weighting matrix
        return sqrt.((1 / N) .* diag(inv(Γ.T * W * Γ)))
    end
end

# Fit models
function fit(X::Matrix{Float64}, Y::Matrix{Float64}, Z::Matrix{Float64}=nothing, type::String="OLS")
    # Initialize
    W = nothing
    N, K = size(X)

    # Check input
    if !(type in ["OLS", "FGLS", "IV", "IVGMM"])
        # Raise error
        error("Please input a valid method (OLS, FGLS, IV, or IVGMM).")
    end

    # Check instruments
    if type in ["IV", "IVGMM"] && Z == nothing
        # Raise error
        error("Please input a matrix for Z if you would like to use instrumental variables.")
    end

    # Check cases
    if type == "IVGMM"
        # Initial weight matrix for IVGMM
        W = inv(Z.T * Z)
    end

    # Estimate parameters
    θ = estimate(X, Y, Z, W)
    
    # Check if GMM
    if type in ["FGLS", "IVGMM"]
        # Estimate weight matrix
        W = weight(X, Y, θ, Z)
    else
        # Return if not GMM
        s = se(X, Y, N, Z)
        return θ, s
    end

    # Re-estimate weighted model
    θ = estimate(X, Y, Z, W)

    # Compute standard errors
    W = weight(X, Y, θ, Z)
    s = se(X, Y, N, Z, W)

    # Return values
    return LGMM(type, θ, s)
end