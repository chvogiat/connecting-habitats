r"""
corridor_budget.py
==================
Rooted terminal-quota corridor design (Extension 1).

A regional agency can rarely afford to connect *every* habitat patch. This
module answers the student-facing question in the case brief -- "If the agency
can fund only part of the network, which segments (and which patches) matter
most?" -- by computing, for each species, the CHEAPEST corridor that connects
the root patch plus at least (p-1) others, for p = 2..|T|.

Model (rooted quota variant of the directed multi-commodity flow in
steiner_exact.py). Root r in T is always connected. For every other terminal k
a binary y_k decides whether k is served; one unit of commodity k is shipped
r -> k only when y_k = 1:

    min   sum_a w_a x_a
    s.t.  flow conservation with demand y_k for each commodity k
          f^k_a <= x_a                         (coupling)
          sum_k y_k >= p - 1                   (connect at least p patches)
          x_a, y_k in {0,1},  f^k_a >= 0

Sweeping p traces the connectivity-vs-cost efficient frontier. At p = |T| the
program must serve every terminal and therefore reduces to the exact Steiner
tree of steiner_exact.py -- a built-in correctness check.

The dual reading -- "connect as many patches as possible for a fixed budget B"
-- is the same frontier read the other way, so the precomputed p-frontier also
drives the budget slider in the interactive dashboard.

Usage:
    python corridor_budget.py            # all species -> budget_frontier.json + fig5
    python corridor_budget.py <species>  # one species (e.g. eastern_box_turtle)
"""
import sys
import json
import time
from pathlib import Path

import pulp

from steiner_exact import solve_status, cbc_solver
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corridor_starter import load_graph, assign_weights, SPECIES_PROFILES

HERE = Path(__file__).parent
JSON_OUT = HERE / "budget_frontier.json"
FIG_OUT = HERE / "fig5_budget_frontier.png"


def _arcs(G):
    arcs, w = [], {}
    for i, j, d in G.edges(data=True):
        arcs.append((i, j)); w[(i, j)] = d["weight"]
        arcs.append((j, i)); w[(j, i)] = d["weight"]
    out_a = {n: [] for n in G.nodes()}
    in_a = {n: [] for n in G.nodes()}
    for (i, j) in arcs:
        out_a[i].append((i, j)); in_a[j].append((i, j))
    return arcs, w, out_a, in_a


def min_cost_connect_p(G, terminals, p, time_limit=120, msg=False):
    """Cheapest corridor connecting the root plus at least (p-1) other patches.

    Returns (edges, cost, served_terminals, status, elapsed).
    """
    r = terminals[0]
    others = terminals[1:]

    # p == 2 is a path from the root to its nearest patch: solve with Dijkstra
    # (the ILP's LP relaxation is weak for cardinality 1, so this is both exact
    # and far faster).
    if p == 2:
        t0 = time.time()
        dist = nx.single_source_dijkstra_path_length(G, r, weight="weight")
        path = nx.single_source_dijkstra_path(G, r, weight="weight")
        k = min(others, key=lambda t: dist[t])
        pth = path[k]
        edges = [[a, b] for a, b in zip(pth, pth[1:])]
        return edges, dist[k], [r, k], "Optimal", time.time() - t0

    arcs, w, out_a, in_a = _arcs(G)

    prob = pulp.LpProblem("budget_steiner", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"x_{a[0]}_{a[1]}", cat="Binary") for a in arcs}
    y = {k: pulp.LpVariable(f"y_{k}", cat="Binary") for k in others}
    f = {(k, a): pulp.LpVariable(f"f_{k}_{a[0]}_{a[1]}", lowBound=0, upBound=1)
         for k in others for a in arcs}

    prob += pulp.lpSum(w[a] * x[a] for a in arcs)
    for k in others:
        for n in G.nodes():
            rhs = y[k] if n == r else (-y[k] if n == k else 0)
            prob += (pulp.lpSum(f[(k, a)] for a in out_a[n])
                     - pulp.lpSum(f[(k, a)] for a in in_a[n])) == rhs
        for a in arcs:
            prob += f[(k, a)] <= x[a]
    prob += pulp.lpSum(y[k] for k in others) >= (p - 1)

    t0 = time.time()
    prob.solve(cbc_solver(msg=msg, timeLimit=time_limit))
    elapsed = time.time() - t0

    edges = [[i, j] for (i, j) in arcs
             if x[(i, j)].value() and x[(i, j)].value() > 0.5]
    served = [r] + [k for k in others
                    if y[k].value() and y[k].value() > 0.5]
    return edges, pulp.value(prob.objective), served, solve_status(prob), elapsed


def _save_row(key, profile, row, terminals):
    """Merge one computed p-row into budget_frontier.json (persist immediately)."""
    data = json.loads(JSON_OUT.read_text()) if JSON_OUT.exists() else {}
    entry = data.get(key, {"description": profile["description"], "rows": []})
    by_p = {r["p"]: r for r in entry["rows"]}
    by_p[row["p"]] = row
    rows = [by_p[p] for p in sorted(by_p)]
    if len(rows) == len(terminals) - 1:            # frontier complete
        full = rows[-1]["cost"]
        for r in rows:
            r["frac_of_full"] = r["cost"] / full
        entry["full_cost"] = full
    entry["rows"] = rows
    data[key] = entry
    JSON_OUT.write_text(json.dumps(data, indent=1))


def make_figure(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"white_tailed_deer": ("#3b7dd8", "o", "White-tailed deer"),
              "eastern_box_turtle": ("#c0392b", "s", "Eastern box turtle"),
              "meadow_vole": ("#2e7d32", "^", "Meadow vole")}
    for key, d in data.items():
        color, marker, label = styles.get(key, ("#555", "o", key))
        xs = [r["frac_of_full"] * 100 for r in d["rows"]]
        ys = [r["p"] for r in d["rows"]]
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                linewidth=2, markersize=7, drawstyle="steps-post")
    ax.set_xlabel("Optimized resistance cost (% of full 8-patch optimum)")
    ax.set_ylabel("Habitat patches connected")
    ax.set_title("Connectivity quota versus optimized resistance cost")
    ax.set_yticks(range(2, 9))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.axvline(100, color="gray", ls="--", lw=1)
    ax.text(100, 2.1, " full network", color="gray", fontsize=9, va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nWrote {FIG_OUT.name}")


def main():
    G, terminals, pos, names = load_graph()
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(terminals)} patches")

    if sys.argv[-1] == "plot":
        make_figure(json.loads(JSON_OUT.read_text()))
        return

    which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in SPECIES_PROFILES else None
    p_lo = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    p_hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(terminals)

    keys = [which] if which else list(SPECIES_PROFILES)
    for key in keys:
        profile = SPECIES_PROFILES[key]
        assign_weights(G, profile)
        print(f"\n{profile['description']}  (p={p_lo}..{p_hi})", flush=True)
        for p in range(p_lo, p_hi + 1):
            edges, cost, served, status, sec = min_cost_connect_p(G, terminals, p)
            row = {"p": p, "cost": cost, "edges": edges, "served": served,
                   "status": status, "sec": round(sec, 2)}
            _save_row(key, profile, row, terminals)
            print(f"  p={p:d}  cost={cost:,.0f}  {sec:.1f}s  served={sorted(served)}",
                  flush=True)

    data = json.loads(JSON_OUT.read_text())
    if len(data) == len(SPECIES_PROFILES) and all(
            len(d["rows"]) == len(terminals) - 1 for d in data.values()):
        make_figure(data)


if __name__ == "__main__":
    main()
