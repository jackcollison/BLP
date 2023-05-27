# Include libraries
using DataFrames, CSV, StatFiles

# Include files
include("Demand.jl")

# Read data
dir = "/Users/jackcollison/Desktop/Wisconsin/Coursework/Second Year/Computational/Q2/PS3/"
chars = DataFrame(load(dir * "Car_demand_characteristics_spec1.dta"));
ins = DataFrame(load(dir * "Car_demand_iv_spec1.dta"));
sims = Float64.(DataFrame(load(dir * "Simulated_type_distribution.dta")).Var1);

# Sort datasets to ensure index lines up
sort!(chars, [:Year, :Model_id])
sort!(ins, [:Year, :Model_id])

# Separate variables
vars = ["Year","price","dpm","hp2wt","size","turbo","trans",
        "model_class_2","model_class_3","model_class_4","model_class_5",
		"cyl_2","cyl_4","cyl_6","cyl_8",
		"drive_2","drive_3"]
ex_vars = ["dpm","hp2wt","size","turbo","trans",
           "model_class_2","model_class_3","model_class_4","model_class_5",
		   "cyl_2","cyl_4","cyl_6","cyl_8",
		   "drive_2","drive_3"]
ins_vars = ["i_import","diffiv_local_0","diffiv_local_1","diffiv_local_2","diffiv_local_3","diffiv_ed_0"]
nl_vars = ["price"]

# Generate data
data = hcat(chars, ins[!, ins_vars])

# Fix Julia bug
BLAS.set_num_threads(1)

# Parameters
markets = data.Year
ν = repeat([sims], size(markets, 1))
θ = [0.6]

# Fit BLP model
demand = @time BLP(data, θ, ins_vars, ex_vars, nl_vars, markets, ν; tol=1e-12, maxiter=1000, verbose=1)

# Compute elasticities
elasticities = ε(demand, data, nl_vars)
OwnElasticities(demand, data, nl_vars; elasticities=elasticities)