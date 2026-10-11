# Merge graph and Procedure ABM (adaptive block merging), Section 6.1.

"""
    merge_violations(I, L, Y, P; rtol = 1e-7, dmin = 1e-6)

Normalized violations of the completion (49) of a solution `Y` of the relaxation with K(P): `V[k, l]` is
the largest value of -Yhat_ab / (Yhat_aa Yhat_bb)^{1/2} over the cross positions (a, b) of the blocks P[k]
and P[l] (zero if all are nonnegative).  Positions with a diagonal product below `dmin` are ignored.
"""
function merge_violations(I::Instance, L::Labels, Y, P; rtol = 1e-7, dmin = 1e-6)
    n, m = I.n, length(P)
    val(a, b) = Y[(min(a, b), max(a, b))]
    W = [val(a, b) for a in 1:n+1, b in 1:n+1]                  # the specified block on (1, x)
    GB = [xblock(I, B) for B in P]
    S = [vcat(1, 1 .+ GB[k]) for k in 1:m]                      # separators S_B
    Bbar = [block_labels(L, P[k]) for k in 1:m]
    Lmat = [[val(a, s) for a in Bbar[k], s in S[k]] * pinv(W[S[k], S[k]]; rtol = rtol) for k in 1:m]
    d(a) = max(val(a, a), 0.0)
    V = zeros(m, m)
    for k in 1:m, l in k+1:m
        E1 = Lmat[k] * W[S[k], 1 .+ GB[l]]                      # Yhat on Bbar_k x G_l
        E2 = Lmat[l] * W[S[l], 1 .+ GB[k]]                      # Yhat on Bbar_l x G_k
        E3 = Lmat[k] * W[S[k], S[l]] * Lmat[l]'                 # Yhat on Bbar_k x Bbar_l
        viol = 0.0
        for (p, a) in enumerate(Bbar[k]), (q, i) in enumerate(GB[l])
            r = sqrt(d(a) * d(1 + i)); r > dmin && (viol = max(viol, -E1[p, q] / r))
        end
        for (p, a) in enumerate(Bbar[l]), (q, i) in enumerate(GB[k])
            r = sqrt(d(a) * d(1 + i)); r > dmin && (viol = max(viol, -E2[p, q] / r))
        end
        for (p, a) in enumerate(Bbar[k]), (q, b) in enumerate(Bbar[l])
            r = sqrt(d(a) * d(b)); r > dmin && (viol = max(viol, -E3[p, q] / r))
        end
        V[k, l] = V[l, k] = viol
    end
    return V
end

"""
    merge_blocks(P, V, tau, kappa)

Step 3 of Procedure ABM: merge along the edges of the merge graph (violation above `tau`) in order of
decreasing violation, skipping merges that would create a block with more than `kappa` groups.  Returns
the new partition, the number of edges, and the number of merges.
"""
function merge_blocks(P, V, tau, kappa)
    m = length(P); parent = collect(1:m); sz = length.(P)
    root(i) = parent[i] == i ? i : (parent[i] = root(parent[i]))
    edges = sort([(V[a, b], a, b) for a in 1:m for b in a+1:m if V[a, b] > tau]; rev = true)
    merged = 0
    for (_, a, b) in edges
        ra, rb = root(a), root(b)
        ra == rb && continue
        if sz[ra] + sz[rb] <= kappa
            parent[rb] = ra; sz[ra] += sz[rb]; merged += 1
        end
    end
    blocks = Dict{Int,Vector{Int}}()
    for k in 1:m
        append!(get!(blocks, root(k), Int[]), P[k])
    end
    return [sort(B) for B in values(blocks)], length(edges), merged
end

"""
    abm(I; kappa = 10, tau = 1e-3, stall = 1e-6, maxsolves = 15, timelimit = 1800.0)

Procedure ABM from the singleton partition P_0.  Returns the history of the passes, with the bound `lb`,
the cumulative solver time `time`, the number of blocks, the largest block, and the reason for stopping
(`"certificate"`: the merge graph has no edges; `"cap"`: no merge is admissible; `"stall"`; `"maxsolves"`;
`"time"`), and the last relaxation solved.
"""
function abm(I::Instance; kappa = 10, tau = 1e-3, stall = 1e-6, maxsolves = 15, timelimit = 1800.0)
    L = Labels(I)
    P = [[j] for j in 1:ngroups(I)]
    hist = NamedTuple[]; total = 0.0; last = nothing
    while true
        r = solve_relaxation(I, L, cliques(I, L, P); timelimit = max(1.0, timelimit - total))
        total += r.time
        if !r.ok
            push!(hist, (lb = NaN, time = total, blocks = length(P), maxblock = maximum(length.(P)), stop = r.status))
            break
        end
        last = r
        V = length(P) >= 2 ? merge_violations(I, L, r.Y, P) : zeros(1, 1)
        newP, nedges, nmerged = length(P) >= 2 ? merge_blocks(P, V, tau, kappa) : (P, 0, 0)
        stalled = !isempty(hist) && r.lb - hist[end].lb <= stall * max(1.0, abs(r.lb))
        stop = nedges == 0 ? "certificate" : nmerged == 0 ? "cap" : stalled ? "stall" :
               length(hist) + 1 >= maxsolves ? "maxsolves" : total >= timelimit ? "time" : ""
        push!(hist, (lb = r.lb, time = total, blocks = length(P), maxblock = maximum(length.(P)), stop = stop))
        stop == "" || break
        P = newP
    end
    return hist, last
end
