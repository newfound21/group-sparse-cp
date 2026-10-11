# Doubly nonnegative (DNN) relaxations of the lifted problems (CP), (CP-signed), and (CP_K).
# Coordinates of the lifted matrix Y: 1 = constant, 1 + i = x_i, 1 + n + j = u_j (inactivity indicator),
# followed by v_j (the slack 1 - u_j: of every group in Burer's reformulation, of the reward groups in
# (CP-signed)) and s_j (the floor slack of a reward group).  A relaxation is given by cliques, index sets
# of coordinates whose principal submatrix is required to be DNN; cliques share their common entries.

"""Coordinates of the lifted problem; `burer = true` adds Burer's slack v_j for every group."""
struct Labels
    n::Int
    G::Int
    v::Dict{Int,Int}               # group j => coordinate of v_j
    s::Dict{Int,Int}               # reward group j => coordinate of s_j
end

function Labels(I::Instance; burer = false)
    n, G, J = I.n, ngroups(I), rewards(I)
    Jv = burer ? collect(1:G) : J
    v = Dict(j => 1 + n + G + k for (k, j) in enumerate(Jv))
    s = Dict(j => 1 + n + G + length(Jv) + k for (k, j) in enumerate(J))
    return Labels(n, G, v, s)
end

nlabels(L::Labels) = 1 + L.n + L.G + length(L.v) + length(L.s)
xlabel(i) = 1 + i
ulabel(L::Labels, j) = 1 + L.n + j
xblock(I::Instance, B) = reduce(vcat, I.groups[B])

"""Labels of the indicator part of the block B: u_j, v_j, and s_j for j in B (the set B-bar)."""
block_labels(L::Labels, B) = vcat([ulabel(L, j) for j in B], [L.v[j] for j in B if haskey(L.v, j)],
                                  [L.s[j] for j in B if haskey(L.s, j)])

"""
    cliques(I, L, P)

Cliques of the relaxation for the partition `P` of the groups: one cone for `P = [1:G]` (the compact
lift, or Burer's reformulation for `L = Labels(I; burer = true)`), and otherwise the star K(P), with a cone
on (1, x) and one cone on (1, x_{G_B}, u_B, ...) for each block B.
"""
cliques(I::Instance, L::Labels, P) = length(P) == 1 ? [collect(1:nlabels(L))] :
    vcat([collect(1:I.n+1)], [vcat(1, 1 .+ xblock(I, B), block_labels(L, B)) for B in P])

"""
    solve_relaxation(I, L, cl; timelimit = 1800.0)

Solve the DNN relaxation with cliques `cl` by MOSEK.  Returns a named tuple with the lower bound `lb`
(the dual objective value plus the constant gamma * sum(a)), the status, the solver time, the entries `Y`
of the solution, and its first-row part `x`.
"""
function solve_relaxation(I::Instance, L::Labels, cl; timelimit = 1800.0)
    n = I.n
    model = Model(Mosek.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", timelimit)
    Y = Dict{Tuple{Int,Int},VariableRef}()
    if length(cl) == 1                           # one cone: a PSD matrix variable
        c = cl[1]; N = length(c)
        @variable(model, Z[1:N, 1:N], PSD)
        for p in 1:N, q in p:N
            Y[(c[p], c[q])] = Z[p, q]
            @constraint(model, Z[p, q] >= 0)
        end
    else                                         # shared nonnegative entries, one PSD constraint per cone
        for c in cl, a in c, b in c
            a <= b && !haskey(Y, (a, b)) && (Y[(a, b)] = @variable(model, lower_bound = 0.0))
        end
        for c in cl
            k = length(c)
            @constraint(model, Symmetric([Y[(min(c[p], c[q]), max(c[p], c[q]))] for p in 1:k, q in 1:k]) in PSDCone())
        end
    end
    y(a, b) = Y[(min(a, b), max(a, b))]
    @constraint(model, y(1, 1) == 1)
    @constraint(model, sum(y(1, xlabel(i)) for i in 1:n) == 1)                       # e'x = 1
    @constraint(model, sum(y(xlabel(i), xlabel(k)) for i in 1:n, k in 1:n) == 1)    # e'Xe = 1
    for (j, g) in enumerate(I.groups)
        u = ulabel(L, j)
        @constraint(model, y(u, u) == y(1, u))                                       # diag(U) = u
        if I.a[j] >= 0                                                               # complementarity
            for i in g
                @constraint(model, y(xlabel(i), u) == 0)
            end
        end
        if haskey(L.v, j)                                                            # u_j + v_j = 1
            v = L.v[j]
            @constraint(model, y(1, u) + y(1, v) == 1)
            @constraint(model, y(u, u) + 2y(u, v) + y(v, v) == 1)
        end
        if I.a[j] < 0                                       # floor: e'x_{G_j} - theta_j v_j - s_j = 0
            v, s = L.v[j], L.s[j]
            idx = vcat([xlabel(i) for i in g], v, s)
            co = vcat(ones(length(g)), -I.theta[j], -1.0)
            @constraint(model, sum(co[p] * y(1, idx[p]) for p in eachindex(idx)) == 0)
            @constraint(model, sum(co[p] * co[q] * y(idx[p], idx[q]) for p in eachindex(idx), q in eachindex(idx)) == 0)
        end
    end
    @objective(model, Min, sum(I.Q[i, k] * y(xlabel(i), xlabel(k)) for i in 1:n, k in 1:n) -
                           I.gamma * sum(I.a[j] * y(1, ulabel(L, j)) for j in 1:L.G))
    t0 = time()
    status = try
        optimize!(model)
        string(termination_status(model))
    catch err
        msg = sprint(showerror, err)
        return (ok = false, status = occursin("1051", msg) ? "OUT_OF_MEMORY" : "ERROR", lb = NaN,
                time = time() - t0, Y = nothing, x = nothing)
    end
    ok = status in ("OPTIMAL", "ALMOST_OPTIMAL", "SLOW_PROGRESS") && has_values(model)
    Yv = ok ? Dict(p => value(w) for (p, w) in Y) : nothing
    return (ok = ok, status = status, lb = ok ? dual_objective_value(model) + I.gamma * sum(I.a) : NaN,
            time = solve_time(model), Y = Yv, x = ok ? [Yv[(1, xlabel(i))] for i in 1:n] : nothing)
end
