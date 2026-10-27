"""Regenerate Figure 1: the frozen synthetic instance with labeled habitat patches.

Kept as a script so that the figure is reproducible from the frozen CSVs like
every other figure in the article.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corridor_starter import load_graph

OUT = "fig1_studyarea.png"


# Drawn at its printed size (0.62 of a 6.5 in text width), so that line widths and
# font sizes below are the sizes in print: candidate lines are 1 pt, the minimum SIAM
# asks for. The patch names are listed in the article's text, not inside the image,
# where they would be too small to read.
WIDTH_IN = 4.0


def main():
    G, terminals, pos, names = load_graph()
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.9 * WIDTH_IN))
    for u, v in G.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.plot([x1, x2], [y1, y2], color="0.72", lw=1.0, zorder=1,
                solid_capstyle="round")
    for idx, t in enumerate(sorted(terminals), start=1):
        x, y = pos[t]
        ax.scatter([x], [y], s=34, color="#a63a28", edgecolors="white",
                   linewidths=1.0, zorder=3)
        ax.annotate(f"P{idx}", (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=8, fontweight="bold",
                    color="#7a2a1c", zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, dpi=300)
    for idx, t in enumerate(sorted(terminals), start=1):
        print(f"P{idx}: {names[t]}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
