"""
real_data_feasibility.py
========================
Does the survivability requirement of Section 4 survive contact with a real road
network?

The generated family of Section 5 answers a question about the *magnitude* of the
premium and cannot answer one about its *feasibility*: every instance in that
family is a perturbed complete lattice, hence 2-edge-connected by construction,
so the requirement is always satisfiable there. Real extracts are not like that.
This script quantifies the difference.

For each place it reports, in order:

  1. structure of the downloaded network: nodes, edges, bridges, articulation
     points, edge connectivity;
  2. whether the requirement is FEASIBLE for the patches as placed, using the
     terminal-aware test -- delete every bridge, then ask whether all patches
     still lie in one component (a bridge into a non-terminal appendage is
     harmless and must not count against feasibility);
  3. the 2-edge-connected core: the largest component left after deleting all
     bridges, and how many patches survive inside it;
  4. the survivability premium on that core for all three species, which is the
     number comparable to the premiums in Tables 5 and 6.

Patches are the placeholder farthest-point samples that acquire_real_data.py
writes. That is a deliberate worst case for step 2 -- farthest-point sampling
puts patches at the extremities of the network, which is exactly where the
dead-end spurs are -- so the script also re-places the patches inside the core
and reports feasibility both ways. Neither placement is protected-area data.

USAGE:
    python real_data_feasibility.py DIR [DIR ...]       # dirs from acquire_real_data.py
    python real_data_feasibility.py --json out.json DIR ...
"""
import csv
import json
import math
import sys
from pathlib import Path

import networkx as nx

from corridor_starter import SPECIES_PROFILES, assign_weights, kou_steiner, total_weight
from steiner_exact import exact_steiner
from corridor_redundant import redundant_corridor

# CBC is not going to prove optimality on an arbitrarily large extract. Above
# this many nodes we report structure only and say so, rather than quietly
# reporting a number from a run that hit its time limit.
SOLVE_MAX_NODES = 1600
TIME_LIMIT = 900


def load_csvs(d):
    """Read a nodes.csv/edges.csv pair without assuming the patch column."""
    d = Path(d)
    G = nx.Graph()
    pos, names, terminals = {}, {}, []
    with open(d / "nodes.csv") as f:
        for r in csv.DictReader(f):
            n = int(r["node_id"])
            G.add_node(n)
            pos[n] = (float(r["lon"]), float(r["lat"]))
            if r["is_habitat_patch"] == "1":
                terminals.append(n)
                names[n] = r["patch_name"]
    with open(d / "edges.csv") as f:
        for r in csv.DictReader(f):
            G.add_edge(int(r["u"]), int(r["v"]),
                       length=float(r["length_m"]), highway=r["highway"])
    return G, terminals, pos, names


def farthest_point(G, pos, k, within=None):
    """The sampler acquire_real_data.py uses, optionally confined to a subset."""
    nodes = sorted(within) if within is not None else sorted(G.nodes())
    if not nodes:
        return []
    picked = [nodes[0]]
    while len(picked) < min(k, len(nodes)):
        best, bestd = None, -1.0
        for c in nodes:
            x, y = pos[c]
            dist = min(math.hypot(x - pos[p][0], y - pos[p][1]) for p in picked)
            if dist > bestd:
                bestd, best = dist, c
        picked.append(best)
    return picked


def terminals_feasible(G, terminals):
    """Terminal-aware test: can every patch still reach every other after any
    single edge is deleted?  Equivalently, do all patches lie in one component
    once every bridge is removed?"""
    if len(terminals) < 2:
        return True, None
    H = G.copy()
    H.remove_edges_from(list(nx.bridges(G)))
    for comp in nx.connected_components(H):
        if set(terminals) <= comp:
            return True, None
    # Name one separating bridge for the report.
    for u, v in nx.bridges(G):
        K = G.copy()
        K.remove_edge(u, v)
        sides = [set(terminals) & c for c in nx.connected_components(K)]
        sides = [s for s in sides if s]
        if len(sides) > 1:
            return False, ((u, v), [sorted(s) for s in sides])
    return False, None


def core_of(G):
    """Largest 2-edge-connected component: delete every bridge, keep the biggest
    piece that is left."""
    H = G.copy()
    H.remove_edges_from(list(nx.bridges(G)))
    comps = [c for c in nx.connected_components(H) if len(c) > 1]
    if not comps:
        return None
    biggest = max(comps, key=len)
    return G.subgraph(biggest).copy()


def study(d, n_patches=8):
    G, terminals, pos, names = load_csvs(d)
    rec = {"place": Path(d).name,
           "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
           "components": nx.number_connected_components(G)}

    # Work on the largest connected component; a disconnected extract is a
    # separate (and more basic) failure mode.
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    rec["giant_nodes"] = giant.number_of_nodes()
    rec["giant_edges"] = giant.number_of_edges()
    rec["bridges"] = len(list(nx.bridges(giant)))
    rec["articulation"] = len(list(nx.articulation_points(giant)))
    rec["bridge_frac"] = rec["bridges"] / max(1, rec["giant_edges"])
    rec["edge_connectivity"] = nx.edge_connectivity(giant) if giant.number_of_nodes() < 900 else None

    # 1. Feasibility for the patches as the acquisition script placed them.
    placed = [t for t in terminals if t in giant]
    rec["patches_placed"] = len(placed)
    ok, why = terminals_feasible(giant, placed)
    rec["feasible_as_placed"] = ok
    if why:
        rec["separating_bridge"] = {"edge": list(why[0]), "sides": why[1]}

    # 2. The 2-edge-connected core.
    core = core_of(giant)
    if core is None:
        rec["core_nodes"] = 0
        return rec
    rec["core_nodes"] = core.number_of_nodes()
    rec["core_edges"] = core.number_of_edges()
    rec["core_node_frac"] = core.number_of_nodes() / giant.number_of_nodes()
    rec["patches_in_core"] = sum(1 for t in placed if t in core)

    # 3. Re-place the patches inside the core and solve there.
    core_terms = farthest_point(core, pos, n_patches, within=set(core.nodes()))
    rec["core_terminals"] = len(core_terms)
    ok2, _ = terminals_feasible(core, core_terms)
    rec["feasible_on_core"] = ok2

    if core.number_of_nodes() > SOLVE_MAX_NODES:
        rec["solved"] = False
        rec["skip_reason"] = f"core has {core.number_of_nodes()} nodes > {SOLVE_MAX_NODES}"
        return rec

    rec["solved"] = True
    rec["species"] = {}
    for key, prof in SPECIES_PROFILES.items():
        assign_weights(core, prof)
        kou = total_weight(kou_steiner(core, core_terms))
        tree, c_star, st1, s1 = exact_steiner(core, core_terms, time_limit=TIME_LIMIT)
        red, c_red, st2, s2 = redundant_corridor(core, core_terms, time_limit=TIME_LIMIT)
        row = {"kou": kou, "exact": c_star, "redundant": c_red,
               "status_exact": st1, "status_redundant": st2,
               "sec_exact": round(s1, 1), "sec_redundant": round(s2, 1)}
        if st1 == "Optimal" and st2 == "Optimal" and c_star:
            row["kou_gap"] = kou / c_star - 1.0
            row["premium"] = c_red / c_star - 1.0
            # Survivability check, done directly rather than trusted.
            bad = 0
            for e in list(red.edges()):
                R = red.copy()
                R.remove_edge(*e)
                if not any(set(core_terms) <= c for c in nx.connected_components(R)):
                    bad += 1
            row["deletion_failures"] = bad
        rec["species"][key] = row
    return rec


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        print(__doc__)
        return 1
    dirs = [d for d in argv if (Path(d) / "nodes.csv").exists()]
    skipped = [d for d in argv if d not in dirs]
    for d in skipped:
        print(f"skipping {d}: no nodes.csv")
    results = []
    for d in dirs:
        print(f"\n=== {d}")
        rec = study(d)
        results.append(rec)
        print(f"  {rec['nodes']} nodes, {rec['edges']} edges, "
              f"{rec['components']} component(s)")
        print(f"  giant: {rec['giant_nodes']} nodes, {rec['bridges']} bridges "
              f"({100*rec['bridge_frac']:.1f}% of edges), "
              f"{rec['articulation']} articulation points")
        print(f"  feasible as placed: {rec['feasible_as_placed']}")
        if rec.get("core_nodes"):
            print(f"  2-edge-connected core: {rec['core_nodes']} nodes "
                  f"({100*rec['core_node_frac']:.0f}% of giant), "
                  f"{rec['patches_in_core']}/{rec['patches_placed']} placed patches inside")
        for k, v in rec.get("species", {}).items():
            if "premium" in v:
                print(f"    {k:20s} gap {100*v['kou_gap']:5.1f}%   "
                      f"premium {100*v['premium']:6.1f}%   "
                      f"deletion failures {v['deletion_failures']}   "
                      f"({v['sec_exact']}s / {v['sec_redundant']}s)")
            else:
                print(f"    {k:20s} NOT SOLVED "
                      f"({v['status_exact']} / {v['status_redundant']})")
        if not rec.get("solved", True):
            print(f"  optimization skipped: {rec['skip_reason']}")
    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=1))
        print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
