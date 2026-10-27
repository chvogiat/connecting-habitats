r"""
generate_instance.py
====================
Generator for the family of stylized corridor instances used in this article.

The frozen teaching instance (nodes.csv / edges.csv) shipped without the code
that produced it, so it could not be varied, resized, or reseeded. That is a
reproducibility gap on its own, and it is also what made the article's headline
percentages impossible to check against anything but a single network. This
script closes both.

Reverse-engineered structure of the frozen instance, recovered from the CSVs:

  * a complete $16\times14$ lattice, node $i$ at (row, col) $=(i//W,\ i\%W)$,
    carrying all $210$ horizontal and $208$ vertical edges;
  * $20$ diagonal shortcuts of the form $i \to i+W+1$;
  * one motorway running the full width of the middle row (row 7, 15 edges),
    which is the barrier the corridor has to cross;
  * two arterial columns (cols 5 and 10, 13 vertical edges each) split between
    trunk and primary;
  * the remaining 397 edges drawn from the minor classes in the proportions
    residential 115, tertiary 76, unclassified 71, track 68, secondary 67;
  * 8 habitat patches placed well apart and off the boundary;
  * node positions jittered off the lattice, with edge length taken from the
    jittered coordinates.

We note that this generator reproduces the *family*, not the shipped instance
bit for bit: the original jitter seed is not recoverable. The frozen CSVs remain
the canonical instance for every number quoted in the article. What the
generator is for is the robustness question, that is, whether those numbers are
properties of the model or of the one network we happened to freeze.

Usage:
    python generate_instance.py --demo            # write one instance and describe it
    python generate_instance.py --sweep 20        # robustness sweep, JSON + summary
"""
import sys
import json
import math
import random
import time
from pathlib import Path

import networkx as nx

HERE = Path(__file__).parent

MINOR_MIX = {"residential": 115, "tertiary": 76, "unclassified": 71,
             "track": 68, "secondary": 67}

# frozen-instance geometry, used as the center of the sweep
BASE = dict(width=16, height=14, n_diag=20, n_patches=8,
            lat0=39.9147, lon0=-88.4673, dlat=0.0349, dlon=0.0356, jitter=0.35)


def generate(seed, width=16, height=14, n_diag=20, n_patches=8,
             lat0=39.9147, lon0=-88.4673, dlat=0.0349, dlon=0.0356,
             jitter=0.35, max_tries=40):
    """Return (G, terminals, pos) for one instance of the family.

    Edge weights are not assigned here; callers apply a species profile with
    assign_weights, exactly as they do for the frozen CSVs.
    """
    rng = random.Random(seed)
    W, H = width, height
    n = W * H

    pos = {}
    for i in range(n):
        r, c = divmod(i, W)
        pos[i] = (lon0 + (c + rng.uniform(-jitter, jitter)) * dlon,
                  lat0 + (r + rng.uniform(-jitter, jitter)) * dlat)

    edges = []
    for i in range(n):
        r, c = divmod(i, W)
        if c + 1 < W:
            edges.append((i, i + 1))
        if r + 1 < H:
            edges.append((i, i + W))
    diag_pool = [i for i in range(n)
                 if (i % W) + 1 < W and (i // W) + 1 < H]
    for i in rng.sample(diag_pool, min(n_diag, len(diag_pool))):
        edges.append((i, i + W + 1))

    # highway classes: one motorway row, two arterial columns, minor elsewhere
    mrow = H // 2
    acols = sorted({max(1, W // 3), min(W - 2, (2 * W) // 3)})
    cls = {}
    for (u, v) in edges:
        r, c = divmod(min(u, v), W)
        horiz, vert = (v - u == 1), (v - u == W)
        if horiz and r == mrow:
            cls[(u, v)] = "motorway"
        elif vert and c in acols:
            cls[(u, v)] = "trunk" if c == acols[0] else "primary"
    minor = [e for e in edges if e not in cls]
    names = list(MINOR_MIX)
    wts = [MINOR_MIX[k] for k in names]
    for e in minor:
        cls[e] = rng.choices(names, weights=wts, k=1)[0]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (u, v) in edges:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        # local equirectangular approximation, meters
        mlat = math.radians((y1 + y2) / 2)
        dx = (x2 - x1) * 111_320 * math.cos(mlat)
        dy = (y2 - y1) * 110_540
        G.add_edge(u, v, length=math.hypot(dx, dy), highway=cls[(u, v)])

    # patches: well separated, off the boundary, and never on the motorway row
    interior = [i for i in range(n)
                if 1 <= i % W <= W - 2 and 1 <= i // W <= H - 2]
    best = None
    for _ in range(max_tries):
        pick = rng.sample(interior, n_patches)
        sep = min(abs(a // W - b // W) + abs(a % W - b % W)
                  for i, a in enumerate(pick) for b in pick[i + 1:])
        if best is None or sep > best[0]:
            best = (sep, pick)
        if sep >= 3:
            break
    terminals = sorted(best[1])
    return G, terminals, pos


# ---------------------------------------------------------------------------
def describe(G, terminals):
    return dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                terminals=len(terminals),
                bridges=len(list(nx.bridges(G))),
                articulation=len(list(nx.articulation_points(G))),
                edge_connectivity=nx.edge_connectivity(G),
                connected=nx.is_connected(G))


def measure(G, terminals, profile, time_limit=300):
    """Independent-links, Kou, exact Steiner, and 2-edge-connected optima."""
    from corridor_starter import assign_weights, kou_steiner, total_weight
    from steiner_exact import exact_steiner
    from corridor_redundant import redundant_corridor

    assign_weights(G, profile)
    closure = nx.Graph()
    for i, s in enumerate(terminals):
        dist = nx.single_source_dijkstra_path_length(G, s, weight="weight")
        for t in terminals[i + 1:]:
            if t in dist:
                closure.add_edge(s, t, weight=dist[t])
    indep = sum(d["weight"] for _, _, d in
                nx.minimum_spanning_tree(closure, weight="weight").edges(data=True))
    kou = total_weight(kou_steiner(G, terminals))
    _, opt, st_o, sec_o = exact_steiner(G, terminals, time_limit=time_limit)
    H2, red, st_r, sec_r = redundant_corridor(G, terminals, time_limit=time_limit)
    return dict(indep=indep, kou=kou, exact=opt, redundant=red,
                kou_gap=kou / opt - 1.0, sharing=1.0 - kou / indep,
                redundancy_premium=red / opt - 1.0,
                status_exact=st_o, status_redundant=st_r,
                sec_exact=sec_o, sec_redundant=sec_r)


def sweep(n_instances=20, seed0=1000, out=HERE / "robustness_sweep.json"):
    from corridor_starter import SPECIES_PROFILES
    rng = random.Random(7)
    rows = []
    for k in range(n_instances):
        cfg = dict(BASE)
        cfg["width"] = rng.choice([14, 15, 16, 17, 18, 20])
        cfg["height"] = rng.choice([12, 13, 14, 15, 16])
        cfg["n_diag"] = rng.choice([10, 15, 20, 25, 30])
        cfg["n_patches"] = rng.choice([6, 7, 8, 9, 10])
        G, terminals, pos = generate(seed0 + k, **cfg)
        d = describe(G, terminals)
        if not d["connected"] or d["edge_connectivity"] < 2:
            print(f"  instance {k}: skipped (edge connectivity "
                  f"{d['edge_connectivity']})")
            continue
        for key, profile in SPECIES_PROFILES.items():
            t0 = time.time()
            m = measure(G, terminals, profile)
            rows.append(dict(instance=k, seed=seed0 + k, species=key,
                             **{c: cfg[c] for c in
                                ("width", "height", "n_diag", "n_patches")},
                             **d, **m))
            print(f"  inst {k:2d} {key:18s} n={d['nodes']:3d} |T|={d['terminals']} "
                  f"kou_gap={m['kou_gap']*100:5.1f}%  "
                  f"redundancy={m['redundancy_premium']*100:5.1f}%  "
                  f"[{m['status_exact']}/{m['status_redundant']}] "
                  f"{time.time()-t0:.0f}s", flush=True)
        Path(out).write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {Path(out).name}  ({len(rows)} rows)")
    return rows


def summarize(rows):
    from corridor_starter import SPECIES_PROFILES
    ok = [r for r in rows if r["status_exact"] == "Optimal"
          and r["status_redundant"] == "Optimal"]
    print(f"\n{len(ok)} of {len(rows)} rows solved to proven optimality\n")
    print(f"{'quantity':24s} {'min':>8s} {'median':>8s} {'max':>8s}  "
          f"{'article claims':>16s}")
    def band(vals):
        # Standard median: the mean of the two middle values when len(v) is even,
        # as reported in the article's table of the generated family.
        v = sorted(vals)
        n = len(v)
        return v[0], (v[(n - 1) // 2] + v[n // 2]) / 2, v[-1]
    for label, key, claim in [("Kou above exact", "kou_gap", "10-14%"),
                              ("sharing saving", "sharing", "2.2-9.4%"),
                              ("redundancy premium", "redundancy_premium", "42-52%")]:
        lo, md, hi = band([r[key] * 100 for r in ok])
        print(f"{label:24s} {lo:7.1f}% {md:7.1f}% {hi:7.1f}%  {claim:>16s}")
    print()
    for key in SPECIES_PROFILES:
        sub = [r for r in ok if r["species"] == key]
        if not sub:
            continue
        lo, md, hi = band([r["redundancy_premium"] * 100 for r in sub])
        print(f"  redundancy premium, {key:20s} {lo:6.1f}% .. {hi:6.1f}% "
              f"(median {md:.1f}%, n={len(sub)})")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        i = sys.argv.index("--sweep")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 20
        summarize(sweep(n))
    else:
        G, T, pos = generate(1000)
        print("one instance of the family:", describe(G, T))
        print("frozen instance for comparison: nodes 224, edges 438, "
              "terminals 8, bridges 0, articulation 0, edge_connectivity 2")
