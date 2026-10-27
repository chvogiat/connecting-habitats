r"""
corridor_consensus_fix.py
=========================
Iterative consensus fixing (Extension 5).

Section 3 of the case study computes a *consensus corridor*: resample the
resistance multipliers, re-solve, and tally how often each segment is chosen.
Segments chosen under almost every draw form a robust backbone.

This script asks the obvious follow-on question. The backbone was computed while
the solver was free to choose *everything*. What if we commit to it -- declare
those segments built -- and re-run the whole Monte-Carlo experiment with them
forced into every solution? Committed segments are sunk cost, so the residual
routing will happily piggyback on them; choices that were previously a coin flip
may now agree. Whatever newly agrees is committed too, and we repeat.

    B <- {}
    repeat
        counts <- MonteCarlo(N draws, every solution forced to contain B)
        new    <- { e not in B : freq(e) >= theta }
        B      <- B u new
    until new = {}

Two facts about this loop, both easy and both worth stating.

TERMINATION.  B never loses an edge and E is finite, so the loop halts after at
most |E| rounds.  It always converges *as a set*.  What it need not converge to
is a corridor: the fixed point B* may be a proper forest, which is the honest
outcome -- a bigger backbone than one round gives, plus an explicit statement of
which routing decisions the data cannot settle.

ACYCLICITY.  Suppose B_t is acyclic. Every draw's solution contains B_t, is
optimal given that, and therefore is itself acyclic (deleting any non-forced
edge of a cycle keeps the terminals connected and strictly lowers cost). If m
newly committed edges closed a cycle with B_t, all m would have to appear
together in some draw: each is absent in at most (1-theta) of the draws, so by
the union bound they co-occur in at least 1 - m(1-theta) of them, which is
positive whenever theta > 1 - 1/m. So for theta > 1 - 1/m no cycle of m new
edges can form; at theta = 0.9 it would take ten simultaneously-committed edges.
The fixed point is a forest, not a mess of cycles, and no threshold above 1/2
can ever commit two edges parallel across the same backbone.

Forcing is implemented by CONTRACTION for the Kou heuristic (each connected
component of B becomes a single super-terminal, and intra-component chords are
dropped since they could only create cycles) and by the constraint
x_ij + x_ji >= 1 for the exact ILP. Both treat every component of B as something
the final corridor must connect, which is what "we committed to building it"
means.

Usage:
    python corridor_consensus_fix.py                       # box turtle, Kou, N=400
    python corridor_consensus_fix.py meadow_vole 600
    python corridor_consensus_fix.py --all                 # all three species
    python corridor_consensus_fix.py --theta-sweep
    python corridor_consensus_fix.py --exact 60            # exact ILP, few draws

Outputs:
    fig9_consensus_fixing.png   -- segments coloured by the round that fixed them
    consensus_fixing_results.json
"""
import sys
import json
import math
import time
import random
import pathlib
from pathlib import Path
from collections import Counter

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from corridor_starter import load_graph, assign_weights, kou_steiner, SPECIES_PROFILES
from corridor_robust import perturbed_weights

HERE = Path(__file__).parent
FIG_OUT = HERE / "fig9_consensus_fixing.png"
FIG_GREEDY = HERE / "fig10_consensus_frontier.png"
JSON_OUT = HERE / "consensus_fixing_results.json"

E = lambda u, v: frozenset((u, v))


# ---------------------------------------------------------------------------
# Forcing a committed edge set B into the solution
# ---------------------------------------------------------------------------
def _components(B):
    """Connected components of the committed edge set, as sets of nodes."""
    g = nx.Graph()
    g.add_edges_from(tuple(e) for e in B)
    return [set(c) for c in nx.connected_components(g)]


def _contract(G, B):
    """Contract every component of B. Returns (Gc, rep, orig).

    rep  maps an original node to its contracted node.
    orig maps a contracted edge (a,b), a<b, back to the original edge it came
    from -- the cheapest one, since parallel edges collapse under contraction.
    Chords inside a component are dropped: they are not forced, and adding one
    to a solution already containing the component could only create a cycle.
    """
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in B:
        u, v = tuple(e)
        parent.setdefault(u, u)
        parent.setdefault(v, v)
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    rep = {n: (find(n) if n in parent else n) for n in G.nodes()}

    Gc = nx.Graph()
    Gc.add_nodes_from(set(rep.values()))
    orig = {}
    for u, v, d in G.edges(data=True):
        if E(u, v) in B:
            continue
        a, b = rep[u], rep[v]
        if a == b:
            continue                      # intra-component chord
        key = (a, b) if a < b else (b, a)
        if not Gc.has_edge(a, b) or d["weight"] < Gc[a][b]["weight"]:
            Gc.add_edge(a, b, weight=d["weight"])
            orig[key] = (u, v)
    return Gc, rep, orig


def kou_forced(G, terminals, B):
    """Kou heuristic constrained to contain B. Returns the edge set (frozensets)."""
    if not B:
        return {E(u, v) for u, v in kou_steiner(G, terminals).edges()}
    Gc, rep, orig = _contract(G, B)
    tc = sorted({rep[t] for t in terminals} | {rep[u] for e in B for u in tuple(e)})
    if len(tc) == 1:                      # backbone already spans everything
        return set(B)
    Hc = kou_steiner(Gc, tc)
    out = set(B)
    for a, b in Hc.edges():
        key = (a, b) if a < b else (b, a)
        u, v = orig[key]
        out.add(E(u, v))
    return out


def exact_forced(G, terminals, B, time_limit=300, msg=False):
    """Exact Steiner-with-committed-edges ILP. Returns (edge set, cost, status, sec).

    Directed multi-commodity flow as in steiner_exact.py, plus
        x_ij + x_ji >= 1   for every committed edge {i,j},
    and one extra commodity per committed component containing no terminal, so
    that the whole of B ends up in one connected corridor.
    """
    import pulp

    from steiner_exact import solve_status, cbc_solver

    r = terminals[0]
    tset = set(terminals)
    extra = [min(c) for c in _components(B) if not (c & tset)]
    commodities = list(terminals[1:]) + extra

    arcs, w = [], {}
    for i, j, d in G.edges(data=True):
        arcs.append((i, j)); w[(i, j)] = d["weight"]
        arcs.append((j, i)); w[(j, i)] = d["weight"]
    out_arcs = {n: [] for n in G.nodes()}
    in_arcs = {n: [] for n in G.nodes()}
    for (i, j) in arcs:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    prob = pulp.LpProblem("steiner_forced", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"x_{a[0]}_{a[1]}", cat="Binary") for a in arcs}
    f = {(k, a): pulp.LpVariable(f"f_{k}_{a[0]}_{a[1]}", lowBound=0, upBound=1)
         for k in commodities for a in arcs}
    prob += pulp.lpSum(w[a] * x[a] for a in arcs)

    for k in commodities:
        for n in G.nodes():
            rhs = 1 if n == r else (-1 if n == k else 0)
            prob += (pulp.lpSum(f[(k, a)] for a in out_arcs[n])
                     - pulp.lpSum(f[(k, a)] for a in in_arcs[n])) == rhs
        for a in arcs:
            prob += f[(k, a)] <= x[a]

    for e in B:
        i, j = tuple(e)
        prob += x[(i, j)] + x[(j, i)] >= 1

    t0 = time.time()
    prob.solve(cbc_solver(msg=msg, timeLimit=time_limit))
    sec = time.time() - t0

    edges = {E(i, j) for (i, j) in arcs
             if x[(i, j)].value() and x[(i, j)].value() > 0.5}
    return edges, pulp.value(prob.objective), solve_status(prob), sec


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def montecarlo_forced(G, terminals, profile, B, n, sigma, rng, exact=False):
    counts = Counter()
    for _ in range(n):
        perturbed_weights(G, profile, sigma, rng)
        H = exact_forced(G, terminals, B)[0] if exact else kou_forced(G, terminals, B)
        counts.update(H)
    return counts


def consensus_fixing(G, terminals, profile, theta=0.90, n=400, sigma=0.35,
                     seed=1, exact=False, max_rounds=25, verbose=True):
    """Iterate to the fixed point under COMMON RANDOM NUMBERS.

    Every round replays the *same* n scenarios (the rng is reseeded identically),
    so a frequency that moves between rounds moved because of the commitment, not
    because we redrew the weather. Without this the round-to-round differences are
    swamped by Monte-Carlo noise of order sqrt(theta(1-theta)/n) ~ 1.5% at n=400.
    """
    B, rounds, fixed_round = set(), [], {}

    for it in range(1, max_rounds + 1):
        rng = random.Random(seed)          # common random numbers
        counts = montecarlo_forced(G, terminals, profile, B, n, sigma, rng, exact)
        freq = {e: c / n for e, c in counts.items() if e not in B}
        new = {e for e, fr in freq.items() if fr >= theta}
        ever = len(freq) + len(B)
        below = sorted((fr for fr in freq.values() if fr < theta), reverse=True)
        bands = {"0.75-theta": sum(1 for f in below if f >= 0.75),
                 "0.50-0.75": sum(1 for f in below if 0.50 <= f < 0.75),
                 "0.25-0.50": sum(1 for f in below if 0.25 <= f < 0.50),
                 "<=0.25": sum(1 for f in below if f < 0.25)}
        rounds.append(dict(round=it, committed_before=len(B), new=len(new),
                           ever_used=ever, bands=bands,
                           runner_up=below[:5],
                           volatile=bands["<=0.25"]))
        if verbose:
            print(f"  round {it}: |B|={len(B):3d}  +{len(new):3d} new  "
                  f"| ever used {ever:3d}  near-miss[0.75,{theta}) "
                  f"{bands['0.75-theta']:2d}  mid[0.5,0.75) "
                  f"{bands['0.50-0.75']:2d}  volatile {bands['<=0.25']:3d}"
                  f"  | top below threshold: "
                  + ", ".join(f"{f:.2f}" for f in below[:4]))
        if not new:
            break
        for e in new:
            fixed_round[e] = it
        B |= new

    return B, rounds, fixed_round, freq


# ---------------------------------------------------------------------------
# Greedy variant: what to do when the fixed point is a forest
# ---------------------------------------------------------------------------
def greedy_consensus(G, terminals, profile, n=300, sigma=0.35, seed=1,
                     exact=False, max_steps=200, price_every=1, verbose=True):
    """Commit the most-agreed uncommitted segment, repeat until B is a corridor.

    The fixed-threshold loop stalls as soon as nothing clears theta. Dropping the
    threshold and committing the single best-supported segment each round instead
    always terminates *connected*: while B fails to join the terminals every draw
    contributes at least one uncommitted edge, so the argmax has positive
    frequency and |B| strictly grows inside a finite E. It therefore traces a
    frontier -- each step reports the support of the segment it just bought and
    what the commitment has cost against the nominal optimum -- and the operator
    picks where to stop rather than being handed a single tree.
    """
    assign_weights(G, profile)
    _, nominal, _, _ = exact_forced(G, terminals, set())
    B, trace = set(), []

    for step in range(1, max_steps + 1):
        d = describe(G, terminals, B)
        if d["is_corridor"]:
            break
        rng = random.Random(seed)
        counts = montecarlo_forced(G, terminals, profile, B, n, sigma, rng, exact)
        freq = {e: c / n for e, c in counts.items() if e not in B}
        if not freq:
            break
        top = max(freq.values())
        chosen = {e for e, fr in freq.items() if fr >= top - 1e-9}
        B |= chosen

        price = None
        if step % price_every == 0:
            assign_weights(G, profile)
            _, con, _, _ = exact_forced(G, terminals, B)
            price = con / nominal - 1.0
        d2 = describe(G, terminals, B)
        trace.append(dict(step=step, committed=len(B), support=top,
                          bought=len(chosen), price=price,
                          components=d2["components"],
                          cycles=d2["independent_cycles"],
                          terminals_joined=d2["terminals_in_one_component"]))
        if verbose:
            pr = "     -" if price is None else f"{price*100:+6.2f}%"
            print(f"  step {step:3d}: |B|={len(B):3d} (+{len(chosen)}) "
                  f"support={top:.2f}  comps={d2['components']:2d}  "
                  f"cycles={d2['independent_cycles']}  "
                  f"joined={d2['terminals_in_one_component']}/{len(terminals)}  "
                  f"price={pr}")
    return B, trace, nominal


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def describe(G, terminals, B):
    g = nx.Graph()
    g.add_edges_from(tuple(e) for e in B)
    comps = list(nx.connected_components(g)) if g.number_of_nodes() else []
    cyc = g.number_of_edges() - g.number_of_nodes() + len(comps) if comps else 0
    tset = set(terminals)
    covered = sum(1 for t in terminals if t in g)
    biggest = max((len(c & tset) for c in comps), default=0)
    return dict(edges=len(B), components=len(comps), independent_cycles=cyc,
                terminals_touched=covered, terminals_in_one_component=biggest,
                is_forest=(cyc == 0),
                is_corridor=(cyc == 0 and len(comps) == 1 and covered == len(terminals)))


def nominal_costs(G, terminals, profile, B):
    """Price of consensus: cheapest corridor containing B vs. unconstrained optimum."""
    assign_weights(G, profile)
    _, opt, st_o, _ = exact_forced(G, terminals, set())
    _, con, st_c, _ = exact_forced(G, terminals, B)
    return dict(nominal_optimum=opt, constrained_optimum=con,
                premium=con / opt - 1.0, status=(st_o, st_c))


# ---------------------------------------------------------------------------
def run_species(G, terminals, key, theta, n, sigma, exact, seed=1):
    profile = SPECIES_PROFILES[key]
    print(f"\n{profile['description']}   theta={theta}  N={n}  sigma={sigma}"
          f"  [{'exact' if exact else 'Kou'}]")
    t0 = time.time()
    B, rounds, fixed_round, freq = consensus_fixing(G, terminals, profile, theta, n,
                                                    sigma, seed, exact)
    d = describe(G, terminals, B)
    cost = nominal_costs(G, terminals, profile, B)
    print(f"  fixed point after {len(rounds)} rounds in {time.time()-t0:.0f}s")
    print(f"  |B*| = {d['edges']}   components = {d['components']}   "
          f"cycles = {d['independent_cycles']}   "
          f"terminals in one component = {d['terminals_in_one_component']}"
          f"/{len(terminals)}")
    print(f"  round-1 backbone {rounds[0]['new']} -> fixed point {d['edges']} "
          f"({d['edges']/max(rounds[0]['new'],1):.2f}x)")
    print(f"  nominal optimum {cost['nominal_optimum']:,.0f}  vs. cheapest "
          f"corridor containing B* {cost['constrained_optimum']:,.0f}  "
          f"(price of consensus {cost['premium']*100:+.2f}%)")
    verdict = ("a full corridor -- the loop closed" if d["is_corridor"] else
               "a forest: %d committed fragments, not a tree" % d["components"])
    print("  outcome: " + verdict)
    return dict(species=key, description=profile["description"], theta=theta,
                n=n, sigma=sigma, exact=exact, rounds=rounds, **d, **{
                    k: v for k, v in cost.items() if k != "status"},
                fixed_round={",".join(map(str, sorted(tuple(e)))): r
                             for e, r in fixed_round.items()},
                envelope={",".join(map(str, sorted(tuple(e)))): round(f, 4)
                          for e, f in freq.items()})


def make_figure(G, terminals, pos, fixed_round, envelope, key, theta, stats):
    """Committed core (thick, by round) against the segments still undecided."""
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.add_collection(LineCollection([[pos[u], pos[v]] for u, v in G.edges()],
                                     colors="0.90", linewidths=0.5, zorder=1))

    # everything the perturbed solutions still disagree about, shaded by support
    env = [(e, f) for e, f in envelope.items() if e not in fixed_round]
    if env:
        cmap_e = plt.get_cmap("YlOrRd")
        ax.add_collection(LineCollection(
            [[pos[tuple(e)[0]], pos[tuple(e)[1]]] for e, _ in env],
            colors=[cmap_e(0.20 + 0.55 * f) for _, f in env],
            linewidths=[0.5 + 1.8 * f for _, f in env], zorder=2))

    rmax = max(fixed_round.values())
    cmap = plt.get_cmap("viridis")
    col = lambda r: cmap(0.0 if rmax == 1 else (r - 1) / (rmax - 1) * 0.85)
    for r in range(1, rmax + 1):
        es = [e for e, rr in fixed_round.items() if rr == r]
        if not es:
            continue
        ax.add_collection(LineCollection(
            [[pos[tuple(e)[0]], pos[tuple(e)[1]]] for e in es],
            colors=[col(r)], linewidths=4.0 - 0.5 * r, zorder=10 + rmax - r))

    ax.scatter([pos[t][0] for t in terminals], [pos[t][1] for t in terminals],
               s=75, c="#8B0000", edgecolors="white", linewidths=1.2, zorder=20)

    handles = [Line2D([0], [0], color=col(r), lw=3,
                      label=f"committed, round {r} "
                            f"({sum(1 for v in fixed_round.values() if v == r)} segs)")
               for r in range(1, rmax + 1)]
    handles.append(Line2D([0], [0], color=plt.get_cmap("YlOrRd")(0.55), lw=1.6,
                          label=f"still undecided ({len(env)} segs, shaded by support)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)
    ax.set_title(
        f"Iterative consensus fixing: {SPECIES_PROFILES[key]['description']}\n"
        f"fixed point after {stats['rounds']} rounds (θ={theta}): "
        f"{stats['edges']} segments committed in {stats['components']} fragments",
        fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nWrote {FIG_OUT.name}")


def make_greedy_figure(trace, key, nominal, out=None):
    """The frontier the greedy variant traces: support bought vs. cost paid."""
    xs = [t["committed"] for t in trace]
    sup = [t["support"] for t in trace]
    pr = [(t["price"] or 0.0) * 100 for t in trace]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.step(xs, sup, where="post", color="#2b6cb0", lw=2,
            label="support of the segment just committed")
    ax.axhline(0.9, color="#2b6cb0", ls=":", lw=1)
    ax.text(xs[-1], 0.892, "θ = 0.90  (where the fixed-threshold loop stalls)",
            fontsize=8, color="#2b6cb0", va="top", ha="right")
    ax.set_xlabel("segments committed")
    ax.set_ylabel("Monte-Carlo support", color="#2b6cb0")
    ax.tick_params(axis="y", labelcolor="#2b6cb0")
    ax.set_ylim(0.42, 1.06)

    ax2 = ax.twinx()
    ax2.step(xs, pr, where="post", color="#c05621", lw=2,
             label="cost above the nominal optimum")
    ax2.set_ylabel("price of consensus (%)", color="#c05621")
    ax2.tick_params(axis="y", labelcolor="#c05621")
    ax2.set_ylim(-0.05, max(1.4, max(pr) * 1.35))

    joined = [(t["committed"], t["terminals_joined"]) for t in trace]
    for i in range(1, len(joined)):
        if joined[i][1] > joined[i - 1][1]:
            ax.axvline(joined[i][0], color="0.78", lw=0.8, zorder=0)
            ax.text(joined[i][0], 1.028, f"{joined[i][1]}", fontsize=7.5,
                    color="0.40", ha="center")
    ax.text(xs[0] - 0.5, 1.028, "patches joined:", fontsize=7.5, color="0.40",
            ha="left")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5, frameon=False)
    ax.set_title(f"Greedy consensus completion: "
                 f"{SPECIES_PROFILES[key]['description']}\n"
                 f"committing the best-supported segment until the corridor closes",
                 fontsize=10.5)
    fig.tight_layout()
    out = FIG_GREEDY if out is None else pathlib.Path(out)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.name}")


def make_stacked_greedy_figure(results, out=None):
    """All three completions on one shared pair of axes, stacked for comparison.

    The price axis is shared deliberately: the deer's premium is more than ten
    times the vole's, and per-panel scaling would hide exactly the contrast the
    figure exists to show. Both axes carry extra headroom at the bottom so that
    the per-panel labels sit in a band no curve reaches.
    """
    pmax = max(max((t["price"] or 0) * 100 for t in r["trace"]) for r in results)
    xmax = max(max(t["committed"] for t in r["trace"]) for r in results)
    fig, axes = plt.subplots(len(results), 1, figsize=(6.8, 5.8), sharex=True)

    for i, (ax, r) in enumerate(zip(axes, results)):
        t = r["trace"]
        xs = [x["committed"] for x in t]
        ax.step(xs, [x["support"] for x in t], where="post", color="#2b6cb0",
                lw=1.9, label="support of the segment just committed")
        ax.axhline(0.9, color="#2b6cb0", ls=":", lw=1)
        ax.set_ylim(0.30, 1.09)
        ax.set_xlim(0, xmax + 1)
        ax.set_yticks([0.5, 0.7, 0.9])
        ax.tick_params(axis="y", labelcolor="#2b6cb0", labelsize=8.5)

        ax2 = ax.twinx()
        ax2.step(xs, [(x["price"] or 0) * 100 for x in t], where="post",
                 color="#c05621", lw=1.9, label="cost above the nominal optimum")
        ax2.set_ylim(-0.28, pmax * 1.12)
        ax2.set_yticks([0.0, 0.5, 1.0, 1.5])
        ax2.tick_params(axis="y", labelcolor="#c05621", labelsize=8.5)

        joined = [(x["committed"], x["terminals_joined"]) for x in t]
        for k in range(1, len(joined)):
            if joined[k][1] > joined[k - 1][1]:
                ax.axvline(joined[k][0], color="0.80", lw=0.7, zorder=0)
                ax.text(joined[k][0], 1.012, f"{joined[k][1]}", fontsize=6.5,
                        color="0.42", ha="center")

        ax.text(0.012, 0.045, r["description"], transform=ax.transAxes,
                fontsize=10, ha="left", style="italic")
        ax.text(0.988, 0.045,
                f"closes at {t[-1]['price']*100:+.2f}% after {len(t)} rounds",
                transform=ax.transAxes, fontsize=9, ha="right",
                color="#c05621")

        if i == 0:
            ax.text(0.012, 1.012, "patches joined:", fontsize=6.5,
                    color="0.42", transform=ax.transAxes)
        if i == 1:
            ax.set_ylabel("Monte-Carlo support", color="#2b6cb0", fontsize=9.5)
            ax2.set_ylabel("price of consensus (%)", color="#c05621", fontsize=9.5)
        if i == len(results) - 1:
            ax.set_xlabel("segments committed", fontsize=9.5)
            ax.tick_params(axis="x", labelsize=8)
            handles = [Line2D([0], [0], color="#2b6cb0", lw=1.9,
                              label="support of the segment just committed"),
                       Line2D([0], [0], color="#c05621", lw=1.9,
                              label="cost above the nominal optimum")]

    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(h_pad=0.9, rect=[0, 0, 1, 0.972])
    out = (FIG_GREEDY.with_name("fig10_consensus_frontier.png")
           if out is None else pathlib.Path(out))
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.name}")


def main():
    argv = sys.argv[1:]
    exact = "--exact" in argv
    do_all = "--all" in argv
    sweep = "--theta-sweep" in argv
    n = (60 if exact else 400)
    theta = 0.90
    keys = ["eastern_box_turtle"]
    for a in argv:
        if a.isdigit():
            n = int(a)
        elif a in SPECIES_PROFILES:
            keys = [a]
        elif a.startswith("--theta="):
            theta = float(a.split("=")[1])
    if do_all:
        keys = list(SPECIES_PROFILES)

    sigma_default = 0.35
    G, terminals, pos, names = load_graph()
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(terminals)} habitat patches")

    results = []
    if "--greedy" in argv:
        for k in keys:
            prof = SPECIES_PROFILES[k]
            print(f"\n{prof['description']}  GREEDY consensus completion  "
                  f"N={n} sigma=0.35  [{'exact' if exact else 'Kou'}]")
            B, trace, nominal = greedy_consensus(G, terminals, prof, n=n,
                                                 sigma=sigma_default, exact=exact)
            d = describe(G, terminals, B)
            print(f"  reached a corridor after {len(trace)} steps: |B|={d['edges']}, "
                  f"cycles={d['independent_cycles']}, "
                  f"final price of consensus "
                  f"{trace[-1]['price']*100 if trace and trace[-1]['price'] is not None else 0:+.2f}%")
            results.append(dict(species=k, description=prof["description"],
                                mode="greedy", n=n, sigma=sigma_default,
                                exact=exact, nominal_optimum=nominal,
                                trace=trace, **d))
        JSON_OUT.with_name("consensus_greedy_results.json").write_text(
            json.dumps(results, indent=2))
        print("\nWrote consensus_greedy_results.json")
        usable = [r for r in results if r["trace"]]
        if len(usable) > 1:
            make_stacked_greedy_figure(usable)
        for r in usable:
            make_greedy_figure(
                r["trace"], r["species"], r["nominal_optimum"],
                FIG_GREEDY.with_name(
                    f"fig10_consensus_frontier_{r['species']}.png"))
        return

    if sweep:
        for th in (0.75, 0.80, 0.90, 0.95, 0.99):
            results.append(run_species(G, terminals, keys[0], th, n, 0.35, exact))
    else:
        for k in keys:
            results.append(run_species(G, terminals, k, theta, n, 0.35, exact))

    JSON_OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {JSON_OUT.name}")

    # figure for the first (or default) species at the nominal threshold
    k = keys[0]
    r0 = next((r for r in results if r["species"] == k and r["theta"] == theta), results[0])
    fr = {frozenset(map(int, e.split(","))): v for e, v in r0["fixed_round"].items()}
    env = {frozenset(map(int, e.split(","))): v for e, v in r0["envelope"].items()}
    if fr:
        make_figure(G, terminals, pos, fr, env, k, r0["theta"],
                    dict(rounds=len(r0["rounds"]), edges=r0["edges"],
                         components=r0["components"]))


if __name__ == "__main__":
    main()
