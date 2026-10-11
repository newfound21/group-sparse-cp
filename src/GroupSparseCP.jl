"""
    GroupSparseCP

Doubly nonnegative relaxations, adaptive block merging (Procedure ABM), and upper bounds for standard
quadratic programs with a weighted group-sparsity term, as in Section 6 of "Compact completely positive
reformulations for quadratic programs with weighted group sparsity".
"""
module GroupSparseCP

using JuMP, MosekTools, Gurobi, LinearAlgebra, Random, Printf

export Instance, instance, ngroups, rewards, objective, clean,
       Labels, cliques, solve_relaxation,
       merge_violations, merge_blocks, abm,
       gurobi_benchmark, dnn_guided_ub

include("instances.jl")
include("relaxations.jl")
include("abm.jl")
include("upper_bounds.jl")

end
