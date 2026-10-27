"""
Ecological Corridor Design — starter analysis
=============================================
Runs entirely on the frozen CSV dataset (no internet required), so results
are identical for every student and instructor. The SAME code runs on live
OpenStreetMap data: see acquire_real_data.py to produce nodes.csv / edges.csv
for any county, state, or country, then point DATA_DIR at the output.

Pipeline:
  1. Load graph + habitat patches (terminals).
  2. Assign species-specific movement-resistance edge weights.
  3. Baseline: minimum spanning tree on the terminal metric closure.
  4. Heuristic Steiner tree (Kou, Markowsky & Berman, 1981).
  5. Compare cost and structure; visualize.
"""
import math
import heapq
import csv
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# Data may ship either flat (nodes.csv next to this file) or in a data/ subdir.
_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data" if (_HERE / "data" / "nodes.csv").exists() else _HERE

# ---------------------------------------------------------------------------
# Species movement-resistance profiles.
# Multiplier applied to physical edge length for each OSM highway class.
# Values are ILLUSTRATIVE for teaching; a research paper must source these
# from the landscape-ecology resistance-surface literature.
# ---------------------------------------------------------------------------
SPECIES_PROFILES = {
    "white_tailed_deer": {
        "description": "White-tailed deer (Odocoileus virginianus)",
        "highway_resistance": {
            "motorway": 10.0, "trunk": 8.0, "primary": 4.0, "secondary": 2.5,
            "tertiary": 1.5, "residential": 1.2, "unclassified": 1.1, "track": 1.0,
        },
        "default_resistance": 1.3,
    },
    "eastern_box_turtle": {
        "description": "Eastern box turtle (Terrapene carolina)",
        "highway_resistance": {
            "motorway": 50.0, "trunk": 40.0, "primary": 25.0, "secondary": 15.0,
            "tertiary": 8.0, "residential": 5.0, "unclassified": 4.0, "track": 2.0,
        },
        "default_resistance": 6.0,
    },
    "meadow_vole": {
        "description": "Meadow vole (Microtus pennsylvanicus)",
        "highway_resistance": {
            "motorway": 30.0, "trunk": 25.0, "primary": 18.0, "secondary": 12.0,
            "tertiary": 7.0, "residential": 4.0, "unclassified": 3.0, "track": 1.5,
        },
        "default_resistance": 4.0,
    },
}


def load_graph(data_dir=DATA_DIR):
    """Return (G, terminals, pos, patch_names) from the frozen CSVs."""
    G = nx.Graph()
    pos, patch_names, terminals = {}, {}, []
    with open(data_dir / "nodes.csv") as f:
        for r in csv.DictReader(f):
            nid = int(r["node_id"])
            G.add_node(nid)
            pos[nid] = (float(r["lon"]), float(r["lat"]))
            if r["is_habitat_patch"] == "1":
                terminals.append(nid)
                patch_names[nid] = r["patch_name"]
    with open(data_dir / "edges.csv") as f:
        for r in csv.DictReader(f):
            G.add_edge(int(r["u"]), int(r["v"]),
                       length=float(r["length_m"]), highway=r["highway"])
    return G, terminals, pos, patch_names


def assign_weights(G, profile):
    res, default = profile["highway_resistance"], profile["default_resistance"]
    for _, _, d in G.edges(data=True):
        d["weight"] = d["length"] * res.get(d["highway"], default)


def kou_stages(G, terminals):
    """One execution of the Kou-Markowsky-Berman heuristic, with every stage kept.

    The four costs of the article, C_ind >= C_union >= C_KMB (>= C*), compare
    stages of ONE run. They must therefore come from one closure tree and one
    set of chosen shortest paths; computing the baseline and the heuristic in
    separate routines lets a tie pick different closure trees in each, and the
    inequality can then fail (C_KMB = 11 > C_union = 9 on Figure 2(b) with
    ad = 4 and terminal order b, c, a, d). Both wrappers below call this.

    Tie handling, fixed and documented so that the output is reproducible:
      * shortest paths: from each terminal s, in the order given, NetworkX's
        Dijkstra keeps the first shortest path it finds to each later terminal t
        (this depends on the graph's edge insertion order);
      * closure MST: Kruskal over the triples (distance, s, t) in sorted order,
        so equal distances are resolved by the node labels of s and then t;
      * cleanup MST: nx.minimum_spanning_tree (Kruskal) on the expanded union,
        whose ties are resolved by the order in which its edges were added;
      * pruning: non-terminal leaves are removed until none remain.

    Returns a dict with the closure tree (list of (distance, s, t)), the chosen
    paths, the expanded union, the final tree, and C_ind, C_union, C_KMB.
    """
    # 1: shortest paths between every terminal pair (the metric closure)
    paths, closure_edges = {}, []
    for i, s in enumerate(terminals):
        dist, path = nx.single_source_dijkstra(G, s, weight="weight")
        for t in terminals[i + 1:]:
            if t in dist:
                closure_edges.append((dist[t], s, t))
                paths[(s, t)] = path[t]
    # 2: MST of the closure via Kruskal + union-find
    parent = {t: t for t in terminals}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    closure_tree = []
    for w, s, t in sorted(closure_edges):
        rs, rt = find(s), find(t)
        if rs != rt:
            parent[rs] = rt
            closure_tree.append((w, s, t))
    # 3: expand each closure edge into its chosen path; count each edge once
    union = nx.Graph()
    for _, s, t in closure_tree:
        p = paths[(s, t)]
        for a, b in zip(p, p[1:]):
            union.add_edge(a, b, weight=G[a][b]["weight"])
    # 4: MST of the expanded subgraph
    steiner = nx.minimum_spanning_tree(union, weight="weight")
    # 5: prune non-terminal leaves
    term = set(terminals)
    changed = True
    while changed:
        changed = False
        for leaf in [n for n in steiner.nodes() if steiner.degree(n) == 1]:
            if leaf not in term:
                steiner.remove_node(leaf)
                changed = True
    return {
        "closure_tree": closure_tree,
        "paths": {(s, t): paths[(s, t)] for _, s, t in closure_tree},
        "union": union,
        "tree": steiner,
        "C_ind": sum(w for w, _, _ in closure_tree),
        "C_union": total_weight(union),
        "C_KMB": total_weight(steiner),
    }


def mst_baseline(G, terminals):
    """Closure MST expanded to roads: the union stage of kou_stages."""
    return kou_stages(G, terminals)["union"]


def kou_steiner(G, terminals):
    """Kou-Markowsky-Berman 2(1-1/|T|)-approximation: the final stage of kou_stages."""
    return kou_stages(G, terminals)["tree"]


def total_weight(H):
    return sum(d["weight"] for _, _, d in H.edges(data=True))


def run_all():
    G, terminals, pos, names = load_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(terminals)} habitat patches\n")
    summary = {}
    for key, profile in SPECIES_PROFILES.items():
        assign_weights(G, profile)
        mst = mst_baseline(G, terminals)
        stn = kou_steiner(G, terminals)
        mc, sc = total_weight(mst), total_weight(stn)
        summary[key] = dict(desc=profile["description"], mst=mc, steiner=sc,
                            ratio=sc / mc, stn_nodes=stn.number_of_nodes(),
                            stn_edges=stn.number_of_edges())
        print(f"{profile['description']}")
        print(f"  MST cost      : {mc:,.0f}")
        print(f"  Steiner cost  : {sc:,.0f}")
        print(f"  Steiner/MST   : {sc/mc:.3f}")
        print(f"  Steiner size  : {stn.number_of_nodes()} nodes, "
              f"{stn.number_of_edges()} edges\n")
    return G, terminals, pos, names, summary


if __name__ == "__main__":
    run_all()
