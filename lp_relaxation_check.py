"""Compare the LP relaxation of the Steiner MCF formulation with the integer optimum.

Reports two distinct things for each species, because Section 3.2 makes two distinct
claims and they are easy to conflate:

  1. the objective gap, (IP - LP)/IP, from solving the relaxation directly; and
  2. the integrality residual of the returned design vector -- the largest distance
     from any x_a to the nearer of 0 and 1.

An objective gap of zero says the bound is tight on this instance. A residual near
zero says the particular vertex CBC returned happened to be integral. Neither says
the relaxation is an integral polyhedron, and Section 3.2 declines to claim that.

Writes lp_relaxation_check.json so the quoted residual has a record behind it.

    python lp_relaxation_check.py
"""
import json
import sys
from pathlib import Path

import pulp

from corridor_starter import load_graph, assign_weights, SPECIES_PROFILES
from steiner_exact import exact_steiner, solve_status, cbc_solver, GAP_REL, THREADS

OUT = Path(__file__).parent / "lp_relaxation_check.json"


def lp_relaxation(G, terminals):
    """Solve the continuous relaxation; return (objective, max integrality residual)."""
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
    p = pulp.LpProblem("lp", pulp.LpMinimize)
    # Continuous in [0,1] instead of binary: this is the original relaxation of the
    # model as written, deliberately kept distinct from whatever bound the solver
    # reports after its own presolve and cut generation.
    x = {a: pulp.LpVariable(f"x_{a[0]}_{a[1]}", lowBound=0, upBound=1) for a in arcs}
    f = {(k, a): pulp.LpVariable(f"f_{k}_{a[0]}_{a[1]}", lowBound=0, upBound=1)
         for k in comm for a in arcs}
    p += pulp.lpSum(w[a] * x[a] for a in arcs)
    for k in comm:
        for n in G:
            rhs = 1 if n == r else (-1 if n == k else 0)
            p += (pulp.lpSum(f[(k, a)] for a in out[n])
                  - pulp.lpSum(f[(k, a)] for a in inn[n])) == rhs
        for a in arcs:
            p += f[(k, a)] <= x[a]
    p.solve(cbc_solver(msg=False, threads=THREADS, gapRel=GAP_REL))

    residual, worst_arc = 0.0, None
    for a, var in x.items():
        v = var.value()
        if v is None:
            continue
        d = min(abs(v - 0.0), abs(v - 1.0))
        if d > residual:
            residual, worst_arc = d, a
    return pulp.value(p.objective), residual, worst_arc, solve_status(p)


def main():
    G, T, pos, names = load_graph()
    records = []
    print(f"{'species':24s} {'LP bound':>13s} {'IP optimum':>13s} {'gap %':>8s} "
          f"{'max |x - nearest int|':>23s}")
    for key, prof in SPECIES_PROFILES.items():
        assign_weights(G, prof)
        lp, residual, worst_arc, status = lp_relaxation(G, T)
        _, ip, ip_status, _ = exact_steiner(G, T)
        gap = 100.0 * (ip - lp) / ip
        print(f"{prof['description'][:24]:24s} {lp:13,.2f} {ip:13,.2f} {gap:8.4f} "
              f"{residual:23.3e}")
        records.append({
            "species": key,
            "lp_objective": lp, "lp_status": status,
            "ip_objective": ip, "ip_status": ip_status,
            "objective_gap_percent": gap,
            "max_integrality_residual": residual,
            "worst_arc": list(worst_arc) if worst_arc else None,
        })
    OUT.write_text(json.dumps(records, indent=1))
    worst = max(r["max_integrality_residual"] for r in records)
    print(f"\nLargest integrality residual over all three relaxations: {worst:.3e}")
    print(f"Wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
