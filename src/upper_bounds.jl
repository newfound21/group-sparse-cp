# Upper bounds: the Gurobi benchmark and the DNN-guided heuristic (Section 6.2).

const GRB_ENV = Ref{Any}(nothing)
grb_env() = (GRB_ENV[] === nothing && (GRB_ENV[] = Gurobi.Env()); GRB_ENV[])

"""
    gurobi_benchmark(I; keep = 1:ngroups(I), timelimit = 600.0, gap = 1e-6)

Gurobi on (MIQPCC) and (MIQPCC-signed) with the complementarity constraints replaced by x_i <= 1 - u_j
(i in G_j, a_j >= 0), which is valid on the simplex, and with the floors e'x_{G_j} >= theta_j (1 - u_j) on the
reward groups.  The groups outside `keep` are switched off.  Returns the objective value `ub` of the
cleaned incumbent, Gurobi's lower bound `lb`, the time, the status, and the incumbent `x`.
"""
function gurobi_benchmark(I::Instance; keep = 1:ngroups(I), timelimit = 600.0, gap = 1e-6)
    n = I.n
    m = Model(() -> Gurobi.Optimizer(grb_env())); set_silent(m)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "TimeLimit", timelimit)
    set_optimizer_attribute(m, "MIPGap", gap)
    set_optimizer_attribute(m, "Threads", 8)
    @variable(m, 0 <= x[1:n] <= 1)
    @variable(m, u[1:ngroups(I)], Bin)
    @constraint(m, sum(x) == 1)
    for (j, g) in enumerate(I.groups)
        if !(j in keep)
            @constraint(m, [i in g], x[i] == 0); @constraint(m, u[j] == 1)
        elseif I.a[j] >= 0
            @constraint(m, [i in g], x[i] <= 1 - u[j])
        else
            @constraint(m, sum(x[i] for i in g) >= I.theta[j] * (1 - u[j]))
        end
    end
    @objective(m, Min, x' * I.Q * x + I.gamma * sum(I.a[j] * (1 - u[j]) for j in 1:ngroups(I)))
    optimize!(m)
    xv = has_values(m) ? clean(value.(x)) : nothing
    return (ub = xv === nothing ? NaN : objective(I, xv), lb = objective_bound(m), time = solve_time(m),
            status = string(termination_status(m)), x = xv)
end

"""
    dnn_guided_ub(I, x; K = 12, timelimit = 60.0)

The heuristic of the upper-bounding remark: rank the groups by the sums of the entries of `x` (the first
row of a relaxation's solution) over the groups, and solve the benchmark formulation restricted to the `K`
top-ranked groups.  Returns the upper bound, the time, and the number of groups in its support.
"""
function dnn_guided_ub(I::Instance, x; K = 12, timelimit = 60.0)
    sums = [sum(max.(x[g], 0.0)) for g in I.groups]
    keep = sortperm(sums; rev = true)[1:min(K, ngroups(I))]
    r = gurobi_benchmark(I; keep = keep, timelimit = timelimit, gap = 1e-9)
    support = r.x === nothing ? -1 : count(any(r.x[i] > 0 for i in g) for g in I.groups)
    return (ub = r.ub, time = r.time, groups = support)
end
