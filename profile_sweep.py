r"""
profile_sweep.py
================
Does the redundancy premium depend on the resistance profile?

The sweep of Section~5 varies the landscape but reuses the same three species
profiles, so its $40$--$91\%$ range is a statement about shapes and not about
assumptions. The resistance multipliers are the one input the article flags as a
guess, which makes them the more interesting thing to vary.

Because the premium is a ratio of two costs computed under the same weights, any
overall scaling of a profile cancels. Only the *shape* can matter, and the
natural one-parameter family for shape is road aversion. We order the eight
highway classes from least to most resistant and set

    r_c = rho^(rank_c / 7),      rank_c = 0 for track ... 7 for motorway,

so that rho = 1 leaves distance-only weights (the road-class penalty is
flattened, though the candidate topology is still the road network) and rho = 100 is an
animal that will almost never cross a motorway. The three species of the article
sit at roughly rho = 10 (deer), 20 (vole), and 25 (turtle).

Sweep A varies rho on a monotone profile. Sweep B draws profiles at random,
without imposing that resistance increase with road class, which separates the
effect of the spread from the effect of the ordering.

Usage:
    python profile_sweep.py            # sweeps A and B, writes profile_sweep.json
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import networkx as nx

from corridor_starter import load_graph
from steiner_exact import exact_steiner
from corridor_redundant import redundant_corridor
from generate_instance import generate

HERE = Path(__file__).parent
OUT = HERE / "profile_sweep.json"

# least to most resistant, matching the classes present in the frozen data
CLASSES = ["track", "unclassified", "residential", "tertiary",
           "secondary", "primary", "trunk", "motorway"]


def monotone_profile(rho):
    n = len(CLASSES) - 1
    return {c: rho ** (i / n) for i, c in enumerate(CLASSES)}


def random_profile(rng, sigma=1.0):
    return {c: math.exp(rng.gauss(0, sigma)) for c in CLASSES}


def apply_profile(G, res, default=1.0):
    for _, _, d in G.edges(data=True):
        d["weight"] = d["length"] * res.get(d["highway"], default)


def premium(G, terminals, res, time_limit=300):
    apply_profile(G, res)
    _, tree, st_t, _ = exact_steiner(G, terminals, time_limit=time_limit)
    _, red, st_r, _ = redundant_corridor(G, terminals, time_limit=time_limit)
    if st_t != "Optimal" or st_r != "Optimal":
        return None
    return dict(tree=tree, redundant=red, premium=red / tree - 1.0)


def spread(res):
    return max(res.values()) / min(res.values())


def is_monotone(res):
    v = [res[c] for c in CLASSES]
    return all(a <= b for a, b in zip(v, v[1:]))


def main():
    rows = []
    frozen = load_graph()[:3]
    instances = [("frozen", frozen[0], frozen[1])]
    for seed in (1000, 1007):
        G, T, _ = generate(seed)
        instances.append((f"gen{seed}", G, T))

    RHOS = [1, 1.5, 2, 3, 5, 8, 12, 20, 35, 50, 100]
    print("Sweep A: monotone road aversion")
    for name, G, T in instances:
        for rho in RHOS:
            t0 = time.time()
            r = premium(G, T, monotone_profile(rho))
            if r is None:
                print(f"  {name:9s} rho={rho:<5g} NOT SOLVED"); continue
            rows.append(dict(sweep="A", instance=name, rho=rho, monotone=True,
                             spread=rho, **r))
            print(f"  {name:9s} rho={rho:<5g} premium={r['premium']*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(rows, indent=2))

    print("\nSweep B: random profiles on the frozen instance")
    rng = random.Random(11)
    G, T = frozen[0], frozen[1]
    for k in range(30):
        res = random_profile(rng)
        t0 = time.time()
        r = premium(G, T, res)
        if r is None:
            print(f"  draw {k:2d} NOT SOLVED"); continue
        rows.append(dict(sweep="B", instance="frozen", draw=k,
                         monotone=is_monotone(res), spread=spread(res),
                         profile={c: round(v, 4) for c, v in res.items()}, **r))
        print(f"  draw {k:2d} spread={spread(res):6.1f} "
              f"monotone={str(is_monotone(res)):5s} "
              f"premium={r['premium']*100:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(rows, indent=2))

    print(f"\nWrote {OUT.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = rows or json.loads(OUT.read_text())
    A = [r for r in rows if r["sweep"] == "A"]
    B = [r for r in rows if r["sweep"] == "B"]
    insts = ["frozen", "gen1000", "gen1007"]
    labels = {"frozen": "frozen instance", "gen1000": "generated instance 1",
              "gen1007": "generated instance 2"}
    colors = {"frozen": "#1f4e79", "gen1000": "#2e8b57", "gen1007": "#a0522d"}

    # ---- Figure: premium against road aversion -----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axvspan(8, 25, color="0.90", zorder=0)
    ax.text(14, 33.2, "range the three species occupy", fontsize=7.5,
            color="0.35", ha="center")
    for i in insts:
        sub = sorted((r["rho"], r["premium"] * 100) for r in A if r["instance"] == i)
        ax.plot([x for x, _ in sub], [y for _, y in sub], "o-", ms=3.4, lw=1.7,
                color=colors[i], label=labels[i])
    # the three real species profiles, placed at their own max/min ratio
    real = [("deer", 10.0, 41.5), ("vole", 20.0, 52.1), ("turtle", 25.0, 52.1)]
    ax.plot([r[1] for r in real], [r[2] for r in real], "D", ms=5.5,
            color="#8B0000", mec="white", mew=0.8, zorder=5,
            label="actual species profiles (frozen)")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"road aversion $\rho$   (ratio of motorway to track resistance)",
                  fontsize=9)
    ax.set_ylabel("redundancy premium (%)", fontsize=9)
    ax.set_ylim(31, 72)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("The premium is large even under distance-only weights "
                 r"($\rho=1$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "fig11_premium_vs_aversion.png", dpi=150)
    print("Wrote fig11_premium_vs_aversion.png")

    # ---- Figure: ordering versus spread ------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axvspan(3, 50, color="0.93", zorder=0)
    for j, i in enumerate(insts):
        sub = sorted((r["spread"], r["premium"] * 100) for r in A if r["instance"] == i)
        ax.plot([x for x, _ in sub], [y for _, y in sub], "o-", ms=3.4, lw=1.6,
                color="#1f4e79", alpha=0.85, zorder=3,
                label=("monotone in road class (3 instances)" if j == 0 else None))
    ax.plot([r["spread"] for r in B], [r["premium"] * 100 for r in B], "o", ms=5,
            mfc="none", mec="#c05621", mew=1.3, zorder=4,
            label="scrambled ordering (frozen)")

    pa = np.array([r["premium"] * 100 for r in A])
    sa = np.array([r["spread"] for r in A])
    pb = np.array([r["premium"] * 100 for r in B])
    sb = np.array([r["spread"] for r in B])
    m, n = (sa >= 3) & (sa <= 50), (sb >= 3) & (sb <= 50)
    ax.hlines(np.median(pa[m]), 3, 50, color="#1f4e79", ls="--", lw=1.3, zorder=5)
    ax.hlines(np.median(pb[n]), 3, 50, color="#c05621", ls="--", lw=1.3, zorder=5)
    ax.text(56, np.median(pa[m]) - 4.5, f"median {np.median(pa[m]):.0f}%", fontsize=7.5,
            color="#1f4e79", va="center")
    ax.text(56, np.median(pb[n]) + 5.0, f"median {np.median(pb[n]):.0f}%", fontsize=7.5,
            color="#c05621", va="center")
    ax.text(40, 172, "matched spread", fontsize=7.5, color="0.35", ha="center")

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.8, 420)
    ax.set_ylim(28, 182)
    ax.set_xlabel("spread of the profile (max resistance / min resistance)", fontsize=9)
    ax.set_ylabel("redundancy premium (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("At the same spread, scrambling the ordering doubles the "
                 "dispersion", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "fig12_profile_ordering.png", dpi=150)
    print("Wrote fig12_profile_ordering.png")


def make_combined_figure(rows=None, out=None):
    """The two profile-sensitivity panels side by side, to fit the page budget.

    The panels carry different y-scales on purpose: panel (b) has to reach 165%
    to show the scrambled tail, and forcing panel (a) onto that scale would flatten
    the aversion trend into a horizontal line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = rows or json.loads(OUT.read_text())
    A = [r for r in rows if r["sweep"] == "A"]
    B = [r for r in rows if r["sweep"] == "B"]
    insts = ["frozen", "gen1000", "gen1007"]
    labels = {"frozen": "frozen instance", "gen1000": "generated 1",
              "gen1007": "generated 2"}
    colors = {"frozen": "#1f4e79", "gen1000": "#2e8b57", "gen1007": "#a0522d"}
    fmt = matplotlib.ticker.ScalarFormatter()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # ---- (a) road aversion --------------------------------------------------
    ax.axvspan(8, 25, color="0.90", zorder=0)
    ax.text(14, 32.6, "species range", fontsize=7, color="0.35", ha="center")
    for i in insts:
        sub = sorted((r["rho"], r["premium"] * 100) for r in A if r["instance"] == i)
        ax.plot([x for x, _ in sub], [y for _, y in sub], "o-", ms=3, lw=1.5,
                color=colors[i], label=labels[i])
    ax.plot([10.0, 20.0, 25.0], [41.5, 52.1, 52.1], "D", ms=5, color="#8B0000",
            mec="white", mew=0.7, zorder=5, label="actual species profiles")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(fmt)
    ax.set_xlabel(r"road aversion $\rho$", fontsize=10)
    ax.set_ylabel("redundancy premium (%)", fontsize=10)
    ax.set_ylim(31, 73)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("(a) distance-only weights at $\\rho=1$", fontsize=9.5)

    # ---- (b) ordering versus spread ----------------------------------------
    bx.axvspan(3, 50, color="0.93", zorder=0)
    for j, i in enumerate(insts):
        sub = sorted((r["spread"], r["premium"] * 100) for r in A if r["instance"] == i)
        bx.plot([x for x, _ in sub], [y for _, y in sub], "o-", ms=3, lw=1.5,
                color="#1f4e79", alpha=0.85, zorder=3,
                label=("monotone in road class" if j == 0 else None))
    bx.plot([r["spread"] for r in B], [r["premium"] * 100 for r in B], "o", ms=4.5,
            mfc="none", mec="#c05621", mew=1.2, zorder=4, label="scrambled ordering")

    pa = np.array([r["premium"] * 100 for r in A]); sa = np.array([r["spread"] for r in A])
    pb = np.array([r["premium"] * 100 for r in B]); sb = np.array([r["spread"] for r in B])
    m, n = (sa >= 3) & (sa <= 50), (sb >= 3) & (sb <= 50)
    bx.hlines(np.median(pa[m]), 3, 50, color="#1f4e79", ls="--", lw=1.2, zorder=5)
    bx.hlines(np.median(pb[n]), 3, 50, color="#c05621", ls="--", lw=1.2, zorder=5)
    bx.text(58, np.median(pa[m]) - 7, f"med {np.median(pa[m]):.0f}%", fontsize=7,
            color="#1f4e79", va="center")
    bx.text(58, np.median(pb[n]) + 8, f"med {np.median(pb[n]):.0f}%", fontsize=7,
            color="#c05621", va="center")
    bx.set_xscale("log")
    bx.set_xticks([1, 3, 10, 30, 100, 300])
    bx.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    bx.set_xlim(0.8, 460)
    bx.set_ylim(28, 178)
    bx.set_xlabel("profile spread (max / min resistance)", fontsize=10)
    bx.set_ylabel("redundancy premium (%)", fontsize=10)
    bx.tick_params(labelsize=8.5)
    bx.legend(fontsize=7.5, frameon=False, loc="upper left")
    bx.set_title("(b) ordering, at matched spread", fontsize=9.5)

    fig.tight_layout(w_pad=1.6)
    out = (HERE / "fig11_profile_sensitivity.png") if out is None else out
    fig.savefig(out, dpi=170)
    print(f"Wrote {out.name}")
