# Include libraries
using DataFrames, CSV, CSVFiles, Random

# Include files
include("Demand.jl")

# Read data
dir = "/Users/jackcollison/Desktop/Research/Projects/Ride-Sharing/The Impact of Chicago's Congestion Tax/Data/"
data = DataFrame(load(dir * "demand.csv"))

# Generate variables
data.delta_iia = log.(data.share) .- log.(1.0 .- data.inside_share)
data.market_id = data.market .* " " .* string.(data.date) .* " " .* string.(data.hour)

# Separate variables
ex_vars = ["miles"]
ins_vars = ["n_rivals", "avg_rival_fare", "avg_rival_miles", "avg_rival_seconds", "avg_rival_mph",
		    "pickup_damage", "pickup_num_units", "pickup_injuries",
			"dropoff_damage", "dropoff_num_units", "dropoff_injuries", 
			"pickup_rain", "pickup_snow", "dropoff_rain", "dropoff_snow",
			"geo_damage", "geo_num_units", "geo_injuries", "geo_rain", "geo_snow"]
nl_vars = ["price"]

# Simulate data
R = 100
Random.seed!(0)
ν = randn(size(unique(data.market_id), 1), R, length(nl_vars))

# Fix Julia bug
BLAS.set_num_threads(1)

# Parameters
θ = [0.6]

# Map products to moments: dpdτ * Δτ
τ = Dict("solo downtown" => 2.28 * 0.672, "pooled downtown" => 0.53 * 0.675, "solo suburb" => 0.53 * 0.321, "pooled suburb" => -0.05 * 1.839)
ds = nothing

# Fit BLP model
demand = @time BLP(data, θ, ins_vars, ex_vars, nl_vars, ν; τ=τ, ds=ds, tol=1e-12, maxiter=1000, verbose=1)
dsdτ(demand, data.Model_id, data[!, nl_vars], data.Year, repeat([ν], size(data.Year, 1)), τ)

# Compute elasticities
elasticities = ε(demand, data, nl_vars)
OwnElasticities(demand, data, nl_vars; elasticities=elasticities)