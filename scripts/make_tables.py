"""LaTeX tables of Section 6 (one per family) from results/results.csv; requires booktabs.

    python scripts/make_tables.py [results/results.csv] [tables/]

Gap of a lower bound LB: 100 (UB* - LB) / |UB*| in percent, where UB* is the best upper bound found.
Bold: per row, the smallest gap of the relaxations, if they differ, and the shortest time with that gap.
Dagger: the DNN-guided upper bound is better than Gurobi's incumbent by more than a relative 1e-5.
"""
import csv, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'results.csv')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'tables')
rows = list(csv.DictReader(open(src, newline='')))
SOLVED = ('OPTIMAL', 'ALMOST_OPTIMAL', 'SLOW_PROGRESS')
MARK = {'TIME_LIMIT': 't.l.', 'OUT_OF_MEMORY': 'o.m.'}


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def gap(ub, lb):
    return max(0.0, 100 * (ub - lb) / abs(ub))


def gap_str(g):
    return r'\(>\)100' if g > 100 else f'{g:.2f}'


def time_str(t):
    return f'{t:.1f}' if t < 10 else f'{t:.0f}'


def bold(s, on):
    return r'\textbf{' + s + '}' if on else s


def relaxation_cells(ub, entries):
    """entries: (lb, time, status) per relaxation; the bold rule is applied to the printed values."""
    shown = [(gap_str(gap(ub, lb)), time_str(t)) if st in SOLVED else None for lb, t, st in entries]
    gaps = [float(g) for g, _ in filter(None, shown)]
    best, differ = (min(gaps), len(set(gaps)) > 1) if gaps else (None, False)
    tbest = min((float(t) for g, t in filter(None, shown) if float(g) == best), default=None)
    cells = []
    for v, (_, _, st) in zip(shown, entries):
        if v is None:
            cells.append(r'\multicolumn{2}{c}{' + MARK.get(st, '--') + '}')
        else:
            g, t = v
            cells.append(f'{bold(g, float(g) == best and differ)} & {bold(t, float(g) == best and float(t) == tbest)}')
    return cells


def row_cells(r, methods):
    ub_dnn, ub_grb = num(r['ub_dnn']), num(r['gurobi_ub'])
    ub = min(ub_dnn, ub_grb)
    dagger = r'\rlap{\(^\dagger\)}' if (ub_grb - ub_dnn) / abs(ub_dnn) > 1e-5 else ''
    entries = [(num(r[m + '_lb']), num(r[m + '_time']), r[m + '_status'] if m + '_status' in r else 'OPTIMAL')
               for m in methods]
    grb_time = 't.l.' if r['gurobi_status'] == 'TIME_LIMIT' else time_str(num(r['gurobi_time']))
    return ([r['seed'], f'{ub:.5f}'.replace('-', r'\(-\)') + dagger] + relaxation_cells(ub, entries)
            + [f"{gap_str(gap(ub, num(r['gurobi_lb'])))} & {grb_time}"])


def table(family, label, caption, methods, heads):
    k = len(methods) + 1
    lines = [r'\begin{table}[tbp]', r'\centering', r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\footnotesize', r'\setlength{\tabcolsep}{3.6pt}', r'\begin{tabular}{rcr' + 'rr' * k + '}', r'\toprule',
             ' & & & ' + ' & '.join(r'\multicolumn{2}{c}{' + h + '}' for h in heads + ['Gurobi']) + r'\\',
             ''.join(r'\cmidrule(lr){%d-%d}' % (4 + 2 * i, 5 + 2 * i) for i in range(k)),
             r'\(n\) & inst. & \(\mathrm{UB}^*\) & ' + ' & '.join(['gap & time'] * k) + r'\\', r'\midrule']
    sizes = sorted({int(r['n']) for r in rows if r['family'] == family})
    for b, n in enumerate(sizes):
        block = sorted((r for r in rows if r['family'] == family and int(r['n']) == n), key=lambda r: int(r['seed']))
        for i, r in enumerate(block):
            lines.append((str(n) if i == 0 else '') + ' & ' + ' & '.join(row_cells(r, methods)) + r'\\')
        if b < len(sizes) - 1:
            lines.append(r'\midrule')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{table}']) + '\n'


CAPTION = (r'Results for the \(\ell_0\) family.  The DNN-guided upper bound equals Gurobi\textquoteright s '
           r'incumbent up to a relative \(10^{-5}\), except where marked \(\dagger\), where it is better.  '
           r'Bold: per row, the smallest gap of the relaxations, if they differ, and the shortest time with '
           r'that gap.  t.l.: time limit (30~min for the relaxations, 600~s for Gurobi); o.m.: out of memory.')
UNSIGNED = (['burer', 'compact', 'singleton', 'abm'], ['Burer', 'compact', 'singleton', 'ABM'])
os.makedirs(out, exist_ok=True)
for name, tex in (('table_l0.tex', table('l0', 'tab:l0', CAPTION, *UNSIGNED)),
                  ('table_group.tex', table('group', 'tab:group', r'Results for the group family; columns and '
                                            r'highlighting as in Table~\ref{tab:l0}.', *UNSIGNED)),
                  ('table_signed.tex', table('signed', 'tab:signed', r'Results for the signed family; columns and '
                                             r'highlighting as in Table~\ref{tab:l0}, where compact denotes the '
                                             r'relaxation of~\eqref{prob:CP_signed}; Burer\textquoteright s relaxation '
                                             r'has no counterpart for this family.',
                                             ['compact', 'singleton', 'abm'], ['compact', 'singleton', 'ABM']))):
    open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n').write(tex)
    print('wrote', os.path.join('tables', name))
