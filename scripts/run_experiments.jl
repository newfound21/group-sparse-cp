# Run the experiments of Section 6 and write one CSV row per instance (columns: see README.md).
#
#   julia --project scripts/run_experiments.jl [families] [sizes] [seeds] [output]
#   julia --project scripts/run_experiments.jl l0,group,signed 60,100,150 1,2,3,4,5 results/results.csv
#
# The environment variable SKIP=burer,compact,gurobi skips methods, e.g., for a quick check.
include(joinpath(@__DIR__, "..", "src", "GroupSparseCP.jl"))
using .GroupSparseCP, LinearAlgebra, Printf

const TLIM = 1800.0          # time limit of each relaxation, and of Procedure ABM in total (s)
const GRB_TLIM = 600.0       # time limit of the Gurobi benchmark (s)

arg(k, default) = length(ARGS) >= k ? ARGS[k] : default
families = Symbol.(split(arg(1, "l0,group,signed"), ","))
sizes = parse.(Int, split(arg(2, "60,100,150"), ","))
seeds = parse.(Int, split(arg(3, "1,2,3,4,5"), ","))
out = arg(4, joinpath(@__DIR__, "..", "results", "results.csv"))
skip = Symbol.(split(get(ENV, "SKIP", ""), ","; keepempty = false))

const COLUMNS = ["family", "n", "seed", "G", "rewards", "lambda_min",
                 "burer_lb", "burer_time", "burer_status", "compact_lb", "compact_time", "compact_status",
                 "singleton_lb", "singleton_time", "abm_lb", "abm_time", "abm_solves", "abm_stop",
                 "ub_dnn", "ub_dnn_time", "ub_dnn_groups", "gurobi_ub", "gurobi_lb", "gurobi_time", "gurobi_status"]
fmt(v) = v isa AbstractFloat ? (isnan(v) ? "" : @sprintf("%.9g", v)) : string(v)
none = (ok = false, status = "NA", lb = NaN, time = NaN, x = nothing)

let I = instance(:signed, 12, 1)                     # compilation run
    r = solve_relaxation(I, Labels(I), cliques(I, Labels(I), [collect(1:ngroups(I))]))
    abm(I); dnn_guided_ub(I, r.x; timelimit = 5.0); gurobi_benchmark(I; timelimit = 5.0)
end

isfile(out) || open(io -> println(io, join(COLUMNS, ",")), out, "w")
for family in families, n in sizes, seed in seeds
    I = instance(family, n, seed)
    G, full = ngroups(I), [collect(1:ngroups(I))]
    burer = family == :signed || :burer in skip ? none :
            solve_relaxation(I, Labels(I; burer = true), cliques(I, Labels(I; burer = true), full); timelimit = TLIM)
    compact = :compact in skip ? none : solve_relaxation(I, Labels(I), cliques(I, Labels(I), full); timelimit = TLIM)
    hist, last = abm(I; timelimit = TLIM)
    # the DNN-guided upper bound starts from the last ABM relaxation, or, in the signed family with
    # n <= 100, from the compact relaxation (as in the paper)
    x = family == :signed && n <= 100 && compact.ok ? compact.x : last === nothing ? nothing : last.x
    ub = x === nothing ? (ub = NaN, time = NaN, groups = -1) : dnn_guided_ub(I, x)
    grb = :gurobi in skip ? (ub = NaN, lb = NaN, time = NaN, status = "NA") : gurobi_benchmark(I; timelimit = GRB_TLIM)
    row = [family, n, seed, G, length(rewards(I)), eigmin(Symmetric(I.Q)),
           burer.lb, burer.time, burer.status, compact.lb, compact.time, compact.status,
           hist[1].lb, hist[1].time, hist[end].lb, hist[end].time, length(hist), hist[end].stop,
           ub.ub, ub.time, ub.groups, grb.ub, grb.lb, grb.time, grb.status]
    open(io -> println(io, join(fmt.(row), ",")), out, "a")
    @printf("%-6s n=%3d seed=%d | Burer %s | compact %s | ABM %.6f (%d solves, %s) | UB %.6f | Gurobi %.6f / %.6f\n",
            family, n, seed, burer.status, compact.status, hist[end].lb, length(hist), hist[end].stop,
            ub.ub, grb.ub, grb.lb)
    flush(stdout)
end
