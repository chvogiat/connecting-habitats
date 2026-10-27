"""Regenerate the two Session-2 figures from the frozen CSVs.

    python make_fig_species.py

Writes:
    fig2_species.png       three EXACT Steiner optima, one per resistance profile
    fig4_exact_vs_kou.png  Kou heuristic against the exact optimum, eastern box turtle

Both figures are drawn from the solvers' own selected-edge sets, computed in this
process, so what is plotted cannot drift from what is reported. Each panel is
labelled with the objective value of the network it shows.

A note on why this script exists. Earlier versions of the package shipped these two
PNGs without any code that wrote them, and MANIFEST.md credited them to
steiner_exact.py, which only prints a table. The fig2_species.png distributed that
way in fact showed the Kou heuristic's trees under the caption "Exact Steiner
corridors" -- the turtle panel was pixel-identical to the heuristic panel of
fig4_exact_vs_kou.png. Regenerating from the solver output is the fix, and keeping
the figure code in the repository is what stops it recurring.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from corridor_starter import (SPECIES_PROFILES, assign_weights, kou_steiner,
                              load_graph, total_weight)
from steiner_exact import exact_steiner

SPECIES_OUT = "fig2_species.png"
COMPARE_OUT = "fig4_exact_vs_kou.png"

BACKDROP = "0.84"
# Both figures are drawn at their printed width (0.98 and 0.85 of a 6.5 in text
# width), so the line widths and font sizes here are the sizes in print. Lines are
# at least 1 pt. The panel titles carry the costs; the captions carry the rest.
FONT = 8
EXACT_COLOR = "#1f4e79"     # the colour Figure 5 uses for a solved design
KOU_COLOR = "#2e6b3f"
TERMINAL = "#8B0000"


def draw(ax, G, pos, H, terminals, color):
    """Background candidate network, the selected subgraph, then the patches."""
    ax.add_collection(LineCollection([[pos[u], pos[v]] for u, v in G.edges()],
                                     colors=BACKDROP, linewidths=1.0, zorder=1))
    ax.add_collection(LineCollection([[pos[u], pos[v]] for u, v in H.edges()],
                                     colors=color, linewidths=2.0, zorder=2))
    ax.scatter([pos[t][0] for t in terminals], [pos[t][1] for t in terminals],
               s=22, c=TERMINAL, edgecolors="white", linewidths=1.0, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.autoscale()


def species_figure(G, terminals, pos, solutions):
    """fig2_species.png -- the exact optimum under each of the three profiles."""
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.6))
    for ax, (key, prof) in zip(axes, SPECIES_PROFILES.items()):
        H, opt = solutions[key]["exact_tree"], solutions[key]["exact"]
        draw(ax, G, pos, H, terminals, EXACT_COLOR)
        ax.set_title(f"{prof['description'].split(' (')[0]}\n"
                     f"$C^{{*}}$ = {opt:,.0f}   ({H.number_of_edges()} segments)",
                     fontsize=FONT)
    fig.tight_layout(pad=0.3)
    fig.savefig(SPECIES_OUT, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {SPECIES_OUT}")


def compare_figure(G, terminals, pos, solutions, key="eastern_box_turtle"):
    """fig4_exact_vs_kou.png -- heuristic against optimum for one species."""
    s = solutions[key]
    gap = 100.0 * (s["kou"] - s["exact"]) / s["exact"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))
    for ax, (H, cost, label, color) in zip(axes, [
            (s["kou_tree"], s["kou"], f"Kou heuristic (+{gap:.1f}%)", KOU_COLOR),
            (s["exact_tree"], s["exact"], "Exact optimum", EXACT_COLOR)]):
        draw(ax, G, pos, H, terminals, color)
        ax.set_title(f"{label}\ncost {cost:,.0f}   ({H.number_of_edges()} segments)",
                     fontsize=FONT)
    fig.tight_layout(pad=0.3)
    fig.savefig(COMPARE_OUT, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {COMPARE_OUT}")


def solve_all():
    G, terminals, pos, names = load_graph()
    solutions = {}
    for key, prof in SPECIES_PROFILES.items():
        assign_weights(G, prof)
        kou_tree = kou_steiner(G, terminals)
        exact_tree, opt, status, sec = exact_steiner(G, terminals)
        # steiner_exact.solve_status returns "Optimal" only when the solver
        # reported a PROVEN optimum. A time-limited incumbent comes back as
        # "Feasible (time limit; ...)" and is refused here, because this figure's
        # panels are captioned as optima and labelled with C*.
        if status != "Optimal":
            raise RuntimeError(f"{key}: solver returned {status!r}; refusing to plot "
                               f"an unproven design as an optimum")
        solutions[key] = {"kou_tree": kou_tree, "kou": total_weight(kou_tree),
                          "exact_tree": exact_tree, "exact": opt, "seconds": sec}
        print(f"{key:22s} Kou {solutions[key]['kou']:12,.2f}  "
              f"exact {opt:12,.2f}  gap {100*(solutions[key]['kou']-opt)/opt:6.2f}%  "
              f"[{status}] {sec:.1f}s")
    return G, terminals, pos, solutions


def main():
    G, terminals, pos, solutions = solve_all()
    species_figure(G, terminals, pos, solutions)
    compare_figure(G, terminals, pos, solutions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
