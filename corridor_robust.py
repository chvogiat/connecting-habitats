r"""
corridor_robust.py
==================
Robustness of the corridor to uncertain resistance values (Extension 2).

The species resistance multipliers are expert guesses, not measurements. A fair
question -- and the case's own guiding prompt, "which single assumption, if
wrong, would most change your recommendation?" -- is how sensitive the corridor
is to those guesses.

We answer it with a Monte-Carlo experiment. For each of N draws we multiply every
highway-class resistance by an independent lognormal shock (median 1, so the
base profile is the expected case), recompute the corridor, and tally how often
each road segment is selected. Segments chosen under almost every perturbation
form a robust "consensus corridor" backbone; segments that appear only
occasionally are the fragile, assumption-driven parts of the plan.

Output:
  * fig6_consensus_corridor.png -- segments shaded/weighted by selection frequency
  * console summary -- backbone (>=90% of draws) vs. volatile (<=25%) segment counts

By default we resample the Kou heuristic (milliseconds per draw, so N can be
large). Pass --exact to resample the exact ILP instead (slower, fewer draws).
"""
import sys
import random
import math
from pathlib import Path
from collections import Counter

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from corridor_starter import (load_graph, kou_steiner, SPECIES_PROFILES)

HERE = Path(__file__).parent
FIG_OUT = HERE / "fig6_consensus_corridor.png"


def perturbed_weights(G, profile, sigma, rng):
    """Assign edge weights using resistance multipliers shocked by lognormal noise."""
    base = dict(profile["highway_resistance"])
    shocked = {k: v * math.exp(rng.gauss(0, sigma)) for k, v in base.items()}
    default = profile["default_resistance"] * math.exp(rng.gauss(0, sigma))
    for _, _, d in G.edges(data=True):
        d["weight"] = d["length"] * shocked.get(d["highway"], default)


def montecarlo(G, terminals, profile, n=400, sigma=0.35, exact=False, seed=1):
    rng = random.Random(seed)
    counts = Counter()
    solve = _exact_edges if exact else None
    for _ in range(n):
        perturbed_weights(G, profile, sigma, rng)
        if exact:
            H = solve(G, terminals)
        else:
            H = kou_steiner(G, terminals)
        for u, v in H.edges():
            counts[frozenset((u, v))] += 1
    return counts


def _exact_edges(G, terminals):
    from steiner_exact import exact_steiner
    H, *_ = exact_steiner(G, terminals)
    return H


def make_figure(G, terminals, pos, counts, n, profile, sigma):
    freq = {e: c / n for e, c in counts.items()}
    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    # faint base network
    seg_bg = [[pos[u], pos[v]] for u, v in G.edges()]
    ax.add_collection(LineCollection(seg_bg, colors="0.85", linewidths=0.5, zorder=1))

    # selected segments, shaded + widened by selection frequency
    cmap = plt.get_cmap("YlOrRd")
    segs, cols, lws = [], [], []
    for e, fr in sorted(freq.items(), key=lambda kv: kv[1]):
        u, v = tuple(e)
        segs.append([pos[u], pos[v]])
        cols.append(cmap(0.25 + 0.75 * fr))
        lws.append(0.6 + 4.2 * fr)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=2))

    tx = [pos[t][0] for t in terminals]
    ty = [pos[t][1] for t in terminals]
    ax.scatter(tx, ty, s=70, c="#8B0000", edgecolors="white",
               linewidths=1.2, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(f"Fraction of {n} perturbations selecting the segment")

    ax.set_title(f"Consensus corridor under uncertain resistance\n"
                 f"{profile['description']}  (lognormal shocks, σ={sigma})",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"Wrote {FIG_OUT.name}")


def main():
    exact = "--exact" in sys.argv
    n = 60 if exact else 400
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
    key = "eastern_box_turtle"
    for a in sys.argv[1:]:
        if a in SPECIES_PROFILES:
            key = a
    profile = SPECIES_PROFILES[key]
    sigma = 0.35

    G, terminals, pos, names = load_graph()
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"{profile['description']}; N={n}; {'exact' if exact else 'Kou'}")

    counts = montecarlo(G, terminals, profile, n=n, sigma=sigma, exact=exact)
    freq = {e: c / n for e, c in counts.items()}
    backbone = [e for e, f in freq.items() if f >= 0.90]
    volatile = [e for e, f in freq.items() if f <= 0.25]
    print(f"segments ever used : {len(freq)}")
    print(f"backbone (>=90%)   : {len(backbone)}")
    print(f"volatile (<=25%)   : {len(volatile)}")
    ever = len(freq)
    core = sum(1 for f in freq.values() if f >= 0.99)
    print(f"near-certain(>=99%): {core}  "
          f"-> {core}/{ever} of ever-used segments are effectively fixed")
    make_figure(G, terminals, pos, counts, n, profile, sigma)


if __name__ == "__main__":
    main()
