# Import required libraries
using Parameters, LinearAlgebra

# Structure for primitives
@with_kw struct Primitives 
    # Primitives
    # TODO: Define primitives
end

# Demand structure
struct Demand
    # Multinomial logit function
    function p(x)
        # Overflow trick
        m = max(0.0, maximum(x))
        n = exp.(x .- m)

        # Return value
        return n ./ (exp.(-m) + sum(n))
    end

    # Choice probability
    function f(x)
        return nothing
    end

    # Simulated shares
    function f(x)
        return nothing
    end

    # Contraction mapping
    function f(x)
        return nothing
    end

    # Objective function
    function f(x)
        return nothing
    end

    # Jacobian
    function f(x)
        return nothing
    end

    # Standard errors
    function f(x)
        return nothing
    end

    # Full model
    function f(x)
        return nothing
    end
end