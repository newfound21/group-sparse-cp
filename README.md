# GroupSparseCP

Code and data for the numerical experiments (Section 6) of

> **Compact completely positive reformulations for quadratic programs with weighted group sparsity**
> Immanuel Bomze, Markus Gabl, and Bo Peng (2026).

The test problems are standard quadratic programs with a weighted group-sparsity term,

$$\min_{x \in \Delta_n} \; x^\top Q x + \gamma \sum_{j} a_j \,[\text{group } j \text{ active}],$$

where Δₙ is the standard simplex and Q is indefinite.  The code computes

* doubly nonnegative (DNN) relaxations of the compact lift (CP), of Burer's reformulation, of the signed
  lift (CP-signed), and of the decomposed lifts (CP_K);
* **Procedure ABM** (adaptive block merging), which starts from the singleton blocks and merges blocks
  along the merge graph of the chordal completion until a certificate of exactness is reached;
* the **DNN-guided upper bound**, which solves the problem restricted to the groups ranked highest by a
  relaxation's solution, and the Gurobi benchmark.

## Layout

```
src/GroupSparseCP.jl        module
src/instances.jl            the l0, group, and signed families; objective
src/relaxations.jl          DNN relaxations (MOSEK)
src/abm.jl                  completion, merge graph, Procedure ABM
src/upper_bounds.jl         Gurobi benchmark and DNN-guided upper bound
scripts/run_experiments.jl  runs the experiments, one CSV row per instance
scripts/make_tables.py      LaTeX tables of the paper from the CSV
results/results.csv         results reported in the paper (45 instances)
```

## Requirements

* Julia ≥ 1.10 (tested with 1.12.5); the versions used are pinned in `Manifest.toml`.
* MOSEK 11 (tested with 11.0.30, through Mosek.jl 11.0.1 and MosekTools 0.15.10) and Gurobi 13 (tested
  with 13.0.0, through Gurobi.jl 1.9.2), with valid licenses; both offer free academic licenses.
* Python 3 (standard library only) for the tables.

## Usage

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project scripts/run_experiments.jl                          # all 45 instances, about 18 h
julia --project scripts/run_experiments.jl group 60 1 out.csv       # one instance
SKIP=burer,gurobi julia --project scripts/run_experiments.jl l0 60 1,2 out.csv   # quick check
python scripts/make_tables.py                                       # writes tables/*.tex
```

The arguments are the families (`l0`, `group`, `signed`), the sizes, the seeds, and the output file; rows
are appended to an existing file.  As a library:

```julia
include("src/GroupSparseCP.jl"); using .GroupSparseCP
I = instance(:group, 60, 1)          # Q, groups, costs a, floors theta
hist, last = abm(I)                  # lower bound hist[end].lb, stopping reason hist[end].stop
ub = dnn_guided_ub(I, last.x)        # upper bound ub.ub
```

## Settings

As in Section 6.2: time limit 30 min for each relaxation and for ABM in total; ABM with at most κ = 10
groups per block, merge-graph tolerance τ = 10⁻³, and stall tolerance δ = 10⁻⁶; the DNN-guided upper bound
restricts the problem to the 12 top-ranked groups and gives Gurobi 60 s; the Gurobi benchmark runs with
`NonConvex=2`, a time limit of 600 s, and a relative gap tolerance of 10⁻⁶.  Entries below 10⁻⁶ of a
solver's solution are set to zero before the objective is evaluated.

## Data

`results/results.csv` has one row per instance (family, n, seed).  Bounds are objective values of the
test problem; times are solver times in seconds; an empty value means that no bound was obtained.

| column | meaning |
|---|---|
| `family`, `n`, `seed` | instance; `instance(family, n, seed)` regenerates it |
| `G`, `rewards` | number of groups and of reward groups (a_j < 0) |
| `lambda_min` | smallest eigenvalue of Q |
| `burer_*`, `compact_*` | DNN bound of Burer's reformulation and of (CP), or (CP-signed): `lb`, `time`, `status` |
| `singleton_lb`, `singleton_time` | first relaxation of ABM (the singleton level) |
| `abm_lb`, `abm_time`, `abm_solves`, `abm_stop` | last bound of ABM, total time, relaxations solved, stopping reason |
| `ub_dnn`, `ub_dnn_time`, `ub_dnn_groups` | DNN-guided upper bound, its time, and the groups in its support |
| `gurobi_ub`, `gurobi_lb`, `gurobi_time`, `gurobi_status` | Gurobi benchmark: incumbent, lower bound, time, status |

Statuses are those of MOSEK and Gurobi (`OPTIMAL`, `SLOW_PROGRESS`, `TIME_LIMIT`), `OUT_OF_MEMORY`, or `NA`
(Burer's relaxation has no counterpart for the signed family).

## Reproducibility

The results were obtained on a laptop with an Intel Core Ultra 9 285H processor and 64 GB of memory under
Windows 11.  The code in this repository is a consolidated version of the scripts that produced them.  It
generates identical instances, and rerunning it on the 15 instances with n = 60 reproduces the singleton
and ABM bounds to all printed digits, the compact bounds and the DNN-guided upper bounds to a relative
4·10⁻⁶, and Burer's bounds to 2·10⁻⁵, within the accuracy of the solvers.  Running times, and the bounds of
time-limited Gurobi runs, depend on the machine.  For the signed family, `ub_dnn_groups` was recomputed
after the main runs with the same rule.

## Citation

```bibtex
@unpublished{GroupSparseCP,
  author = {Bomze, Immanuel and Gabl, Markus and Peng, Bo},
  title  = {Compact completely positive reformulations for quadratic programs with weighted group sparsity},
  year   = {2026},
  note   = {Code and data: https://github.com/newfound21/group-sparse-cp}
}
```

## License and acknowledgment

MIT License (see `LICENSE`).  The code was consolidated, tested, and documented with the assistance of
Claude Opus 5.5 (Anthropic).
