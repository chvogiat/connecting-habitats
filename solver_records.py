"""
solver_records.py
=================
Record, for each exact solve reported on the frozen instance, the evidence a reader needs
to judge the solver claim: status, incumbent objective, the gap and node count CBC reports,
the tolerances and thread count it ran under, the runtime convention, and an independent
recomputation of the objective from the selected edges.

    python solver_records.py            # writes solver_records.json

The point of the last column is that a solver's reported objective and the cost of the
design it returns are two different numbers, and only one of them is what the article
quotes. They should agree; this script checks that they do.
"""
import json
import re
import sys
import time
from pathlib import Path

import pulp

from corridor_starter import SPECIES_PROFILES, assign_weights, load_graph
from steiner_exact import GAP_REL, THREADS, solve_status, cbc_solver

HERE = Path(__file__).parent
OUT = HERE / "solver_records.json"
LOG = HERE / "_cbc_log.txt"

# GAP_REL and THREADS are imported from steiner_exact so that this evidence file
# describes the configuration the reported runs actually used, rather than a
# second configuration that happens to be written here.
TIME_LIMIT = 900


def _cbc(log_path):
    return cbc_solver(msg=True, logPath=str(log_path), threads=THREADS,
                      gapRel=GAP_REL, timeLimit=TIME_LIMIT)


def _parse_log(path):
    """Pull the fields CBC prints at the end of a run."""
    txt = Path(path).read_text(errors="replace") if Path(path).exists() else ""
    out = {}
    m = re.search(r"Result - (.+)", txt)
    if m:
        out["cbc_result"] = m.group(1).strip()
    m = re.search(r"Objective value:\s*([-\d.eE+]+)", txt)
    if m:
        out["cbc_objective"] = float(m.group(1))
    m = re.search(r"Gap:\s*([-\d.eE+]+)", txt)
    if m:
        out["cbc_gap"] = float(m.group(1))
    m = re.search(r"Enumerated nodes:\s*(\d+)", txt)
    if m:
        out["cbc_nodes"] = int(m.group(1))
    m = re.search(r"Total iterations:\s*(\d+)", txt)
    if m:
        out["cbc_iterations"] = int(m.group(1))
    m = re.search(r"Lower bound:\s*([-\d.eE+]+)", txt)
    if m:
        out["cbc_lower_bound"] = float(m.group(1))
    return out


def steiner(G, terminals):
    """Directed multi-commodity flow Steiner model, built inline so the solver
    options above are the ones actually used."""
    r = terminals[0]
    comm = terminals[1:]
    arcs, w = [], {}
    for i, j, d in G.edges(data=True):
        arcs += [(i, j), (j, i)]
        w[(i, j)] = d["weight"]
        w[(j, i)] = d["weight"]
    out = {n: [] for n in G}
    inn = {n: [] for n in G}
    for a in arcs:
        out[a[0]].append(a)
        inn[a[1]].append(a)
    p = pulp.LpProblem("steiner", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"x_{a[0]}_{a[1]}", cat="Binary") for a in arcs}
    f = {(k, a): pulp.LpVariable(f"f_{k}_{a[0]}_{a[1]}", lowBound=0, upBound=1)
         for k in comm for a in arcs}
    p += pulp.lpSum(w[a] * x[a] for a in arcs)
    for k in comm:
        for nd in G:
            rhs = 1 if nd == r else (-1 if nd == k else 0)
            p += (pulp.lpSum(f[(k, a)] for a in out[nd])
                  - pulp.lpSum(f[(k, a)] for a in inn[nd])) == rhs
        for a in arcs:
            p += f[(k, a)] <= x[a]
    return p, x, w, None


def survivable(G, terminals):
    """Two units per commodity, joint directional cap, undirected design variables."""
    r = terminals[0]
    comm = terminals[1:]
    edges = [(min(i, j), max(i, j)) for i, j, _ in G.edges(data=True)]
    w = {e: G[e[0]][e[1]]["weight"] for e in edges}
    arcs = [(i, j) for e in edges for (i, j) in (e, (e[1], e[0]))]
    out = {n: [] for n in G}
    inn = {n: [] for n in G}
    for a in arcs:
        out[a[0]].append(a)
        inn[a[1]].append(a)
    p = pulp.LpProblem("survivable", pulp.LpMinimize)
    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat="Binary") for e in edges}
    f = {(k, a): pulp.LpVariable(f"f_{k}_{a[0]}_{a[1]}", lowBound=0, upBound=1)
         for k in comm for a in arcs}
    p += pulp.lpSum(w[e] * x[e] for e in edges)
    for k in comm:
        for nd in G:
            rhs = 2 if nd == r else (-2 if nd == k else 0)
            p += (pulp.lpSum(f[(k, a)] for a in out[nd])
                  - pulp.lpSum(f[(k, a)] for a in inn[nd])) == rhs
        for e in edges:
            p += f[(k, e)] + f[(k, (e[1], e[0]))] <= x[e]
    return p, x, w, edges


def run(name, build, G, terminals):
    if LOG.exists():
        LOG.unlink()
    # Timed from before the model is built, so that "includes model construction"
    # below is a description of the measurement and not merely of the intent.
    t0 = time.time()
    p, x, w, edges = build(G, terminals)
    t_built = time.time()
    p.solve(_cbc(LOG))
    elapsed = time.time() - t0
    build_seconds = t_built - t0
    status = solve_status(p)
    obj = pulp.value(p.objective)
    # Independent recomputation: cost of the design the solver actually returned.
    sel = [k for k, v in x.items() if v.value() is not None and v.value() > 0.5]
    recomputed = sum(w[k] for k in sel)
    rec = {
        "model": name,
        "status": status,
        "pulp_problem_status": pulp.LpStatus[p.status],
        "pulp_solution_status": pulp.LpSolution[getattr(p, "sol_status", 0)],
        "objective_reported": obj,
        "objective_recomputed_from_selected": recomputed,
        "objective_agrees": abs(obj - recomputed) < 1e-6 * max(1.0, abs(obj)),
        "selected": len(sel),
        "wall_seconds": round(elapsed, 2),
        "build_seconds": round(build_seconds, 2),
        "solve_call_seconds": round(elapsed - build_seconds, 2),
        "runtime_convention": ("wall clock from before model construction to after the "
                               "solver call returns; includes model construction, the LP "
                               "file written for CBC, and reading its solution back"),
        "gapRel": GAP_REL, "threads": THREADS, "timeLimit": TIME_LIMIT,
        "solver": "CBC via PuLP %s" % pulp.__version__,
    }
    rec.update(_parse_log(LOG))
    return rec


def main():
    G, terminals, pos, names = load_graph()
    records = []
    for key, prof in SPECIES_PROFILES.items():
        assign_weights(G, prof)
        for name, build in (("steiner", steiner), ("survivable", survivable)):
            r = run(name, build, G, terminals)
            r["species"] = key
            records.append(r)
            print(f"{key:20s} {name:11s} {r['status']:10s} "
                  f"obj={r['objective_reported']:12,.1f} "
                  f"recomputed={r['objective_recomputed_from_selected']:12,.1f} "
                  f"agree={r['objective_agrees']} "
                  f"gap={r.get('cbc_gap', 'n/a')} nodes={r.get('cbc_nodes', 'n/a')} "
                  f"{r['wall_seconds']}s (build {r['build_seconds']}s "
                  f"+ solve {r['solve_call_seconds']}s)")
    if LOG.exists():
        LOG.unlink()
    OUT.write_text(json.dumps(records, indent=1))
    print(f"\nWrote {OUT.name}")
    bad = [r for r in records if not r["objective_agrees"] or r["status"] != "Optimal"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
