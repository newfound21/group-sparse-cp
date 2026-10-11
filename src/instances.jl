# Test problems of Section 6.2:
#     min  x'Qx + gamma * sum_j a_j * [group j active]   over the standard simplex,
# with Q = (R + R')/2 + 3I and R_ik ~ U[-1, 1].  Group j is active if x_{G_j} != 0 when a_j >= 0, and if
# e'x_{G_j} >= theta_j (its activation floor) when a_j < 0; the objective is then Phi_theta of (23).

struct Instance
    family::Symbol                 # :l0, :group or :signed
    n::Int
    seed::Int
    Q::Matrix{Float64}
    groups::Vector{Vector{Int}}    # partition of 1:n into consecutive index sets
    a::Vector{Float64}             # activation costs; the weights w in the l0 and group families
    theta::Vector{Float64}         # activation floors, used for the reward groups (a_j < 0)
    gamma::Float64
end

ngroups(I::Instance) = length(I.groups)
rewards(I::Instance) = [j for j in eachindex(I.a) if I.a[j] < 0]      # the reward groups J_<

# random group sizes in lo:hi that sum to n, without a trailing singleton
function group_sizes(rng, n; lo = 2, hi = 5)
    s = Int[]
    while sum(s) < n
        r = n - sum(s)
        k = r <= hi ? r : rand(rng, lo:hi)
        r - k == 1 && (k -= 1)
        push!(s, max(k, 1))
    end
    return s
end
consecutive_groups(sz) = (c = cumsum(sz); [collect(c[j]-sz[j]+1:c[j]) for j in eachindex(sz)])

"""
    instance(family, n, seed; shift = 3.0, gamma = 0.05, reward_fraction = 0.25, reward = 0.5, theta = 0.1)

Instance `seed` of size `n` of the family `:l0`, `:group` or `:signed`.  The weights are
w_j = |G_j|^{1/2}.  In the signed family, a random quarter of the groups of the group family carry the
reward a_j = -w_j/2 and the activation floor theta_j.  For fixed `n` and `seed`, all families share Q,
and the group and signed families share the groups.
"""
function instance(family::Symbol, n::Int, seed::Int; shift = 3.0, gamma = 0.05,
                  reward_fraction = 0.25, reward = 0.5, theta = 0.1)
    rng = MersenneTwister(7919 * n + seed)
    R = 2rand(rng, n, n) .- 1
    Q = Matrix(Symmetric((R + R') / 2 + shift * LinearAlgebra.I))
    if family == :l0
        groups = [[i] for i in 1:n]
    elseif family in (:group, :signed)
        groups = consecutive_groups(group_sizes(rng, n))
    else
        throw(ArgumentError("unknown family $family"))
    end
    a = sqrt.(length.(groups))
    if family == :signed
        G = length(groups)
        J = randperm(rng, G)[1:max(1, round(Int, reward_fraction * G))]
        a[J] .*= -reward
    end
    return Instance(family, n, seed, Q, groups, a, fill(theta, length(groups)), gamma)
end

"""Objective value at a point `x` of the simplex (the floor convention on the reward groups)."""
function objective(I::Instance, x)
    v = x' * I.Q * x
    for (j, g) in enumerate(I.groups)
        active = I.a[j] >= 0 ? any(x[i] > 0 for i in g) : sum(x[g]) >= I.theta[j] - 1e-6
        active && (v += I.gamma * I.a[j])
    end
    return v
end

"""Project a solver output onto the simplex, setting entries below `tol` (solver tolerances) to zero."""
clean(x; tol = 1e-6) = (y = max.(x, 0.0); y[y .< tol] .= 0.0; y ./ sum(y))
