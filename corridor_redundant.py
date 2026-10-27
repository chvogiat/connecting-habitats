r"""
Redundant (2-edge-connected) corridor networks: the price of not being a tree.
=============================================================================

Motivation
----------
The Steiner tree of ``steiner_exact.py`` is the cheapest way to make every
habitat patch *reachable*.  It is also, by construction, the *most fragile*
way: a tree has no cycles, so every one of its segments is a bridge.  Cut one
road segment -- a new interchange, a widened highway, a washed-out culvert --
and the corridor network splits in two.

Conservation-planning models that optimize connectivity alone share this blind
spot.  Wang, Sefair and Fletcher Jr. (2026) make the point on grid landscapes:
connectivity is only one of several *geometric* requirements a usable corridor
must satisfy, alongside width, bounded size, and *redundancy* -- the presence of
multiple alternative routes between the same pair of patches (McRae et al.,
2008; Pinto and Keitt, 2009).  Width is a raster notion and does not transfer
to a road network, but redundancy transfers exactly: it is 2-edge-connectivity.

Formulation
-----------
Fix a root r in T.  Undirected edge variables x_e in {0,1} select segments;
for every other patch k in T\{r} we ship *two* units of commodity k from r to
k, and cap the total flow of each commodity across an edge at that edge's
capacity:

    min   sum_e w_e x_e
    s.t.  sum_{out of i} f^k_ij - sum_{into i} f^k_ji = +2 if i=r, -2 if i=k, 0 otherwise
          f^k_ij + f^k_ji <= x_e            for every edge e={i,j}, every k
          x_e in {0,1},  f >= 0

By max-flow/min-cut, a feasible 2-unit flow under unit edge capacities exists
exactly when the selected subgraph contains two edge-disjoint r-k paths.  Since
2-edge-connectivity is an equivalence relation on nodes, requiring it only to
the root makes *every* pair of patches doubly connected.

The diff from the Steiner tree model is two lines: the right-hand side 1 becomes
2, and the per-arc coupling f <= x becomes a per-edge coupling summing both
directions.  That is the whole point for the classroom -- a minimal change in
the model, a large change in the map.
"""
import time
import pulp
import networkx as nx

from corridor_starter import (load_graph, assign_weights, total_weight,
                              SPECIES_PROFILES)
from steiner_exact import exact_steiner, solve_status, cbc_solver, GAP_REL, THREADS


def redundant_corridor(G, terminals, time_limit=900, msg=False):
    """Minimum-weight 2-edge-connected Steiner network spanning `terminals`."""
    # Timed from here, including model construction: same convention as
    # steiner_exact.exact_steiner. See the note there.
    t0 = time.time()
    r = terminals[0]
    commodities = terminals[1:]

    edges = [(min(i, j), max(i, j)) for i, j, _ in G.edges(data=True)]
    w = {e: G[e[0]][e[1]]["weight"] for e in edges}

    # Arc view for the flow variables; each arc maps back to its undirected edge.
    arcs, edge_of = [], {}
    for (i, j) in edges:
        arcs += [(i, j), (j, i)]
        edge_of[(i, j)] = (i, j)
        edge_of[(j, i)] = (i, j)

    out_arcs = {n: [] for n in G.nodes()}
    in_arcs = {n: [] for n in G.nodes()}
    for a in arcs:
        out_arcs[a[0]].append(a)
        in_arcs[a[1]].append(a)

    prob = pulp.LpProblem("corridor_2ec", pulp.LpMinimize)
    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat="Binary") for e in edges}
    f = {(k, a): pulp.LpVariable(f"f_{k}_{a[0]}_{a[1]}", lowBound=0, upBound=1)
         for k in commodities for a in arcs}

    prob += pulp.lpSum(w[e] * x[e] for e in edges)

    for k in commodities:
        for n in G.nodes():
            rhs = 2 if n == r else (-2 if n == k else 0)
            prob += (pulp.lpSum(f[(k, a)] for a in out_arcs[n])
                     - pulp.lpSum(f[(k, a)] for a in in_arcs[n])) == rhs
        for (i, j) in edges:
            prob += f[(k, (i, j))] + f[(k, (j, i))] <= x[(i, j)]

    prob.solve(cbc_solver(msg=msg, timeLimit=time_limit,
                          threads=THREADS, gapRel=GAP_REL))
    elapsed = time.time() - t0

    H = nx.Graph()
    for e in edges:
        if x[e].value() and x[e].value() > 0.5:
            H.add_edge(e[0], e[1], weight=w[e])
    return H, pulp.value(prob.objective), solve_status(prob), elapsed


def fragility(H, terminals):
    """(#bridges whose removal severs patches, worst-case patches severed)."""
    term = set(terminals)
    critical, worst = 0, 0
    for u, v in list(nx.bridges(H)):
        wt = H[u][v]["weight"]
        H.remove_edge(u, v)
        comps = list(nx.connected_components(H))
        H.add_edge(u, v, weight=wt)
        sizes = sorted((len(term & c) for c in comps), reverse=True)
        severed = sum(sizes[1:])
        if severed:
            critical += 1
            worst = max(worst, severed)
    return critical, worst


def worst_cut(H, terminals):
    """The bridge of H whose removal severs the most patches."""
    term = set(terminals)
    best, best_sev = None, -1
    for u, v in list(nx.bridges(H)):
        wt = H[u][v]["weight"]
        H.remove_edge(u, v)
        sizes = sorted((len(term & c) for c in nx.connected_components(H)),
                       reverse=True)
        H.add_edge(u, v, weight=wt)
        severed = sum(sizes[1:])
        if severed > best_sev:
            best, best_sev = (u, v), severed
    return best, best_sev


def make_figure(G, terminals, pos, tree, red, profile, premium, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cut, severed = worst_cut(tree.copy(), terminals)
    # Drawn at its printed width (0.98 of a 6.5 in text width): line widths and font
    # sizes are the sizes in print, and every line is at least 1 pt.
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))

    # Which patches end up on each side once the worst segment is deleted.
    _t = tree.copy()
    _w = _t[cut[0]][cut[1]]["weight"]
    _t.remove_edge(*cut)
    _sides = [sorted(set(terminals) & c) for c in nx.connected_components(_t)]
    _sides = [s for s in _sides if s]
    _t.add_edge(cut[0], cut[1], weight=_w)
    _idx = {t: i + 1 for i, t in enumerate(sorted(terminals))}
    _group = {}
    for gi, side in enumerate(sorted(_sides, key=len, reverse=True)):
        for t in side:
            _group[t] = gi

    for ax, (H, title) in zip(axes, [
            (tree, f"Steiner tree: each of its {tree.number_of_edges()}\n"
                   f"segments is a single point of failure"),
            (red, f"Survivable corridor (+{premium*100:.0f}% cost):\n"
                  f"no single segment loss disconnects patches")]):
        ax.add_collection(LineCollection(
            [[pos[u], pos[v]] for u, v in G.edges()],
            colors="0.84", linewidths=1.0, zorder=1))
        ax.add_collection(LineCollection(
            [[pos[u], pos[v]] for u, v in H.edges()],
            colors="#1f4e79", linewidths=2.0, zorder=2))
        ax.scatter([pos[t][0] for t in terminals], [pos[t][1] for t in terminals],
                   s=22, c="#8B0000", edgecolors="white", linewidths=1.0, zorder=4)
        for t in terminals:
            ax.annotate(f"P{_idx[t]}", pos[t], textcoords="offset points",
                        xytext=(3, 2), fontsize=7, color="#5a1010",
                        fontweight="bold", zorder=5)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.autoscale()

    # Mark the worst cut on the tree panel only.
    u, v = cut
    axes[0].add_collection(LineCollection(
        [[pos[u], pos[v]]], colors="white", linewidths=5.0, zorder=3))
    axes[0].add_collection(LineCollection(
        [[pos[u], pos[v]]], colors="#e6194b", linewidths=3.0,
        linestyles=(0, (1.6, 1.0)), zorder=3.1))
    _lab = "  |  ".join(
        "{" + ", ".join(f"P{_idx[t]}" for t in side) + "}"
        for side in sorted(_sides, key=len, reverse=True))
    axes[0].annotate(f"deleting this segment splits\n{_lab}",
                     xy=((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2),
                     xytext=(0.5, 0.03), textcoords="axes fraction", fontsize=7,
                     ha="center", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                               alpha=0.9),
                     color="#e6194b", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#e6194b", lw=1.0))
    # Distinguish each side's patches by BOTH marker shape and fill, so the split
    # survives grayscale printing and colour-blind readers.
    for gi, side in enumerate(sorted(_sides, key=len, reverse=True)):
        axes[0].scatter([pos[t][0] for t in side], [pos[t][1] for t in side],
                        s=[24, 30][gi % 2], marker=["o", "s"][gi % 2],
                        c=["#8B0000", "#f2f2f2"][gi % 2],
                        edgecolors=["white", "#111111"][gi % 2],
                        linewidths=[1.0, 1.0][gi % 2], zorder=4)

    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300)
    print(f"Wrote {out}")


if __name__ == "__main__":
    G, terminals, pos, names = load_graph()
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(terminals)} patches; graph is 2-edge-connected: "
          f"{len(list(nx.bridges(G))) == 0}\n")

    hdr = (f"{'Species':22s} {'Tree':>11s} {'Redundant':>11s} {'Premium':>8s} "
           f"{'segs T':>7s} {'segs R':>7s} {'crit T':>7s} {'crit R':>7s} {'sec':>7s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for key, profile in SPECIES_PROFILES.items():
        assign_weights(G, profile)
        tree, tcost, _, _ = exact_steiner(G, terminals)
        red, rcost, status, sec = redundant_corridor(G, terminals)

        crit_t, worst_t = fragility(tree.copy(), terminals)
        crit_r, worst_r = fragility(red.copy(), terminals)

        if key == "eastern_box_turtle":
            make_figure(G, terminals, pos, tree, red, profile,
                        rcost / tcost - 1, "fig8_redundancy.png")

        rows.append(dict(key=key, desc=profile["description"], tree=tcost,
                         red=rcost, premium=rcost / tcost - 1,
                         segs_t=tree.number_of_edges(),
                         segs_r=red.number_of_edges(),
                         crit_t=crit_t, crit_r=crit_r,
                         worst_t=worst_t, sec=sec, status=status))
        r = rows[-1]
        print(f"{r['desc'][:22]:22s} {tcost:11,.0f} {rcost:11,.0f} "
              f"{r['premium']*100:7.1f}% {r['segs_t']:7d} {r['segs_r']:7d} "
              f"{crit_t:7d} {crit_r:7d} {sec:7.1f}  [{status}]")

    print("\nWorst-case fragmentation of the Steiner tree "
          "(patches severed by the single worst cut):")
    for r in rows:
        print(f"  {r['desc'][:34]:34s} {r['worst_t']} of {len(terminals)} patches")

    import json
    with open("redundancy_results.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote redundancy_results.json")
