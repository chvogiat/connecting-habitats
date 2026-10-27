"""
check_cost.py
=============
Regression tests for the cost accounting of Session 1 and for the two hand-checkable
examples. Everything here is small enough to verify by enumeration or by direct
recomputation, which is the point: these are assertions that the implemented models mean
what the article says, not further experiments.

    python check_cost.py

Checks, in order:

  T1  Tiny example 1 (star/triangle): C* = 3, C_KMB = 3.8, cheapest terminal-survivable
      design = 5.7, by exhaustive enumeration of all 2^6 edge subsets.
  T2  Tiny example 2, the tie-break example of the article's Figure 2(b), with ad = 3 as
      drawn: C* = 9; six minimum closure trees of weight C_ind = 12; over EVERY expansion,
      two trees give (C_union, C_KMB) = (12, 9) and four give (10, 10), as in Table 3.
      The shipped heuristic returns (12, 10, 10) for all 24 terminal orders. Terminal
      survivability is infeasible because b and c each have one incident edge.
      Perturbations: ad = 2 gives six trees of weight 11, branches (11, 9) and (9, 9), so
      C_KMB = 9 in every branch; ad = 4 makes d(a, d) = 4 along two shortest paths, so
      step 3 becomes tie-dependent as well.
  T2c Stage consistency: C_ind >= C_union >= C_KMB for one execution of kou_stages, for
      all 24 terminal orders at ad = 2, 3, 4. This includes the ad = 4 order b, c, a, d,
      on which separately computed baseline and heuristic once disagreed (9 versus 11).
  T3  Frozen instance: C_ind, C_union and C_KMB for all three species, taken from one
      execution and checked against an independent recomputation of C_ind.
  T4  Generated family: all 60 instance-profile runs regenerated from their recorded seeds,
      Kou costs matched against robustness_sweep.json, and the cleanup term computed so that
      the "sharing saving" label in the sweep table is verified rather than assumed.
  T6  Solver status guard: steiner_exact.solve_status must not call a time-limited
      incumbent "Optimal". Checked against the installed PuLP CBC solution-file parser,
      because this is a property of the wrapper version, not of our own code.
"""
import itertools
import json
import sys
from pathlib import Path

import networkx as nx

from corridor_starter import (SPECIES_PROFILES, assign_weights, kou_stages,
                              kou_steiner, load_graph, mst_baseline, total_weight)

HERE = Path(__file__).parent
TOL = 1e-6
failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def indep_links(G, terminals):
    """C_ind: the closure MST measured in the closure, not in the graph."""
    closure = nx.Graph()
    for i, s in enumerate(terminals):
        dist = nx.single_source_dijkstra_path_length(G, s, weight="weight")
        for t in terminals[i + 1:]:
            if t in dist:
                closure.add_edge(s, t, weight=dist[t])
    T = nx.minimum_spanning_tree(closure, weight="weight")
    return sum(d["weight"] for _, _, d in T.edges(data=True))


def enumerate_optima(G, terminals):
    """(cheapest connected design, cheapest design surviving any one of its own
    edge deletions), by brute force over edge subsets."""
    E = list(G.edges())
    best_conn, best_surv = None, None
    for r in range(1, len(E) + 1):
        for sub in itertools.combinations(E, r):
            H = nx.Graph()
            H.add_edges_from(sub)
            if not all(t in H for t in terminals):
                continue
            comp = nx.node_connected_component(H, terminals[0])
            if not set(terminals) <= comp:
                continue
            cost = sum(G[u][v]["weight"] for u, v in sub)
            if best_conn is None or cost < best_conn - TOL:
                best_conn = cost
            ok = True
            for e in sub:
                H2 = H.copy()
                H2.remove_edge(*e)
                if not any(set(terminals) <= c for c in nx.connected_components(H2)):
                    ok = False
                    break
            if ok and (best_surv is None or cost < best_surv - TOL):
                best_surv = cost
    return best_conn, best_surv


def t1_star_triangle():
    print("\nT1  tiny example 1: star, triangle, ring")
    G = nx.Graph()
    for t in "abc":
        G.add_edge("s", t, weight=1.0)
    for u, v in itertools.combinations("abc", 2):
        G.add_edge(u, v, weight=1.9)
    term = list("abc")
    c_star, c_red = enumerate_optima(G, term)
    c_kmb = total_weight(kou_steiner(G, term))
    check("C* = 3", abs(c_star - 3.0) < TOL, f"got {c_star}")
    check("C_KMB = 3.8", abs(c_kmb - 3.8) < TOL, f"got {c_kmb}")
    check("survivable optimum = 5.7", abs(c_red - 5.7) < TOL, f"got {c_red}")


def _fig2b(ad=3.0):
    """Figure 2(b): Steiner node s, spokes a, c, d of weight 2, b of weight 3, edge ad."""
    G = nx.Graph()
    G.add_weighted_edges_from([("s", "a", 2.0), ("s", "b", 3.0), ("s", "c", 2.0),
                               ("s", "d", 2.0), ("a", "d", float(ad))])
    return G, ["a", "b", "c", "d"]


def _clean(U, term):
    M = nx.minimum_spanning_tree(U, weight="weight")
    changed = True
    while changed:
        changed = False
        for n in [x for x in M.nodes() if M.degree(x) == 1 and x not in term]:
            M.remove_node(n)
            changed = True
    return M


def all_branches(G, term):
    """Every minimum closure tree and, for each, the (C_union, C_KMB) of EVERY choice
    of shortest path for its edges. Nothing is dropped when shortest paths tie."""
    D, P = {}, {}
    for x, y in itertools.combinations(term, 2):
        D[(x, y)] = nx.shortest_path_length(G, x, y, weight="weight")
        P[(x, y)] = list(nx.all_shortest_paths(G, x, y, weight="weight"))
    cl = nx.Graph()
    for (x, y), d in D.items():
        cl.add_edge(x, y, weight=d)
    c_ind = nx.minimum_spanning_tree(cl, weight="weight").size(weight="weight")
    out = {}
    for tree in nx.SpanningTreeIterator(cl, weight="weight"):
        if tree.size(weight="weight") > c_ind + TOL:
            break
        key = tuple(sorted("".join(sorted(e)) for e in tree.edges()))
        for choice in itertools.product(*[P[tuple(sorted(e))] for e in tree.edges()]):
            U = nx.Graph()
            for p in choice:
                for s, t in zip(p, p[1:]):
                    U.add_edge(s, t, weight=G[s][t]["weight"])
            out.setdefault(key, set()).add(
                (round(U.size(weight="weight"), 6),
                 round(_clean(U, term).size(weight="weight"), 6)))
    return c_ind, out, P


def t2_tiebreak():
    print("\nT2  tiny example 2: the tie-break example (Figure 2(b))")
    G, term = _fig2b(3)
    c_star, c_red = enumerate_optima(G, term)
    check("C* = 9", abs(c_star - 9.0) < TOL, f"got {c_star}")
    c_ind, br, _ = all_branches(G, term)
    for k in sorted(br):
        print(f"        closure MST {{{','.join(k)}}}:  C_ind = {c_ind:.0f}  "
              f"(C_union, C_KMB) = {sorted(br[k])}")
    check("C_ind = 12 for every minimum closure tree", abs(c_ind - 12.0) < TOL)
    check("exactly six minimum closure trees", len(br) == 6, f"got {len(br)}")
    opt = {k for k, v in br.items() if v == {(12.0, 9.0)}}
    rest = {k for k, v in br.items() if v == {(10.0, 10.0)}}
    check("{ab,ad,cd} and {ac,ad,bd} give (12, 9)",
          opt == {("ab", "ad", "cd"), ("ac", "ad", "bd")}, f"got {sorted(opt)}")
    check("the other four give (10, 10)", len(rest) == 4, f"got {len(rest)}")
    ours = set()
    for order in itertools.permutations(term):
        st = kou_stages(G, list(order))
        ours.add((round(st["C_ind"], 6), round(st["C_union"], 6), round(st["C_KMB"], 6)))
    check("shipped heuristic returns (12, 10, 10) for all 24 terminal orders",
          ours == {(12.0, 10.0, 10.0)}, f"got {sorted(ours)}")
    pendant = sorted(t for t in term if G.degree(t) == 1)
    check("b and c, and only they, are pendant terminals", pendant == ["b", "c"],
          f"got {pendant}")
    check("terminal survivability infeasible", c_red is None,
          "" if c_red is None else f"got {c_red}")

    print("\n    perturbation ad = 2")
    G, term = _fig2b(2)
    c_ind, br, _ = all_branches(G, term)
    vals = sorted(sorted(v) for v in br.values())
    print(f"        C_ind = {c_ind:.0f}; branches {vals}")
    check("six trees of weight 11", len(br) == 6 and abs(c_ind - 11.0) < TOL)
    check("branches are (11, 9) twice and (9, 9) four times",
          vals == [[(9.0, 9.0)]] * 4 + [[(11.0, 9.0)]] * 2, f"got {vals}")
    check("C_KMB = 9 in every branch",
          all(kmb == 9.0 for v in br.values() for _, kmb in v))

    print("\n    perturbation ad = 4")
    G, term = _fig2b(4)
    _, br, P = all_branches(G, term)
    check("d(a, d) = 4 along two distinct shortest paths", len(P[("a", "d")]) == 2,
          f"got {P[('a', 'd')]}")
    outcomes = sorted({x for v in br.values() for x in v})
    print(f"        (C_union, C_KMB) over all trees and expansions: {outcomes}")


def t2c_stage_consistency():
    print("\nT2c stage consistency within one execution, all terminal orders")
    bad = []
    for ad in (2, 3, 4):
        G, term = _fig2b(ad)
        for order in itertools.permutations(term):
            st = kou_stages(G, list(order))
            if not (st["C_ind"] >= st["C_union"] - TOL >= st["C_KMB"] - 2 * TOL):
                bad.append((ad, order, st["C_ind"], st["C_union"], st["C_KMB"]))
            # the wrappers must report stages of that same execution
            if (abs(total_weight(mst_baseline(G, list(order))) - st["C_union"]) > TOL or
                    abs(total_weight(kou_steiner(G, list(order))) - st["C_KMB"]) > TOL):
                bad.append((ad, order, "wrapper mismatch"))
    st = kou_stages(_fig2b(4)[0], ["b", "c", "a", "d"])
    print(f"        ad = 4, order b,c,a,d: C_ind = {st['C_ind']:.0f}  "
          f"C_union = {st['C_union']:.0f}  C_KMB = {st['C_KMB']:.0f}")
    check("C_ind >= C_union >= C_KMB in all 72 executions", not bad, f"{bad[:3]}")


def t3_frozen():
    print("\nT3  frozen instance: the three costs of one execution")
    G, terminals, pos, names = load_graph()
    for key, prof in SPECIES_PROFILES.items():
        assign_weights(G, prof)
        st = kou_stages(G, terminals)
        c_ind, c_union, c_kmb = st["C_ind"], st["C_union"], st["C_KMB"]
        check(f"{key}: C_ind matches an independent recomputation",
              abs(c_ind - indep_links(G, terminals)) < 1e-6)
        print(f"        {key:20s} C_ind={c_ind:12,.1f}  C_union={c_union:12,.1f}  "
              f"C_KMB={c_kmb:12,.1f}  cleanup={c_union - c_kmb:.3f}")
        check(f"{key}: C_ind >= C_union >= C_KMB",
              c_ind >= c_union - TOL and c_union >= c_kmb - TOL)
        check(f"{key}: cleanup term is zero", abs(c_union - c_kmb) < 1e-3)


def t4_generated():
    print("\nT4  generated family: 60 runs regenerated from recorded seeds")
    path = HERE / "robustness_sweep.json"
    if not path.exists():
        print("        robustness_sweep.json not found; skipping")
        return
    from generate_instance import generate
    rows = json.load(open(path))
    byinst = {}
    for r in rows:
        byinst.setdefault(r["instance"], r)
    mismatched, cleanup_nonzero, n = 0, 0, 0
    for k in sorted(byinst):
        cfg = byinst[k]
        G, terminals, _ = generate(cfg["seed"], width=cfg["width"], height=cfg["height"],
                                   n_diag=cfg["n_diag"], n_patches=cfg["n_patches"])
        if G.number_of_nodes() != cfg["nodes"]:
            mismatched += 1
            continue
        for key, prof in SPECIES_PROFILES.items():
            assign_weights(G, prof)
            c_union = total_weight(mst_baseline(G, terminals))
            c_kmb = total_weight(kou_steiner(G, terminals))
            rec = next(r for r in rows if r["instance"] == k and r["species"] == key)
            n += 1
            if abs(c_kmb - rec["kou"]) > 1e-6:
                mismatched += 1
            if c_union - c_kmb > 1e-6:
                cleanup_nonzero += 1
    check(f"all {n} runs regenerate and match the stored Kou cost", mismatched == 0,
          f"{mismatched} mismatches")
    check("cleanup term is zero in every run, so the sweep's "
          "'sharing saving' really is sharing", cleanup_nonzero == 0,
          f"{cleanup_nonzero} runs with a positive cleanup term")


def t5_percentiles():
    """The percentile ranks quoted for the frozen instance's heuristic gaps."""
    print("\nT5  percentile ranks of the frozen gaps within the generated family")
    path = HERE / "robustness_sweep.json"
    if not path.exists():
        print("        robustness_sweep.json not found; skipping")
        return
    rows = json.load(open(path))
    gaps = sorted(r["kou_gap"] for r in rows)
    n = len(gaps)
    # Percentile rank = fraction of runs STRICTLY BELOW the value, in percent.
    frozen = {"meadow_vole": 0.1008, "white_tailed_deer": 0.1214,
              "eastern_box_turtle": 0.1391}
    expected = {"meadow_vole": 85, "white_tailed_deer": 88, "eastern_box_turtle": 93}
    for key in ("meadow_vole", "white_tailed_deer", "eastern_box_turtle"):
        g = frozen[key]
        rank = round(100.0 * sum(1 for x in gaps if x < g) / n)
        print(f"        {key:20s} gap {100*g:5.2f}%  -> percentile rank {rank}")
        check(f"{key} ranks at {expected[key]}", rank == expected[key], f"got {rank}")
    import math
    p90 = gaps[math.ceil(0.9 * n) - 1]
    print(f"        90th percentile of the 60 runs = {100*p90:.1f}%")
    check("only the turtle's gap is in the top decile",
          frozen["eastern_box_turtle"] > p90 >= frozen["white_tailed_deer"])


def t6_status_guard():
    """A time-limited CBC run that still holds an incumbent must not read as optimal.

    PuLP 2.7.0 maps the CBC header "Stopped on time - objective value X" to
    LpStatusOptimal, so the problem status alone cannot distinguish it from a proven
    optimum; only the solution status can. This test drives the *installed* parser with
    representative headers, so it will also catch a future PuLP whose behaviour differs.
    """
    import os
    import tempfile
    import pulp
    from pulp.apis.coin_api import COIN_CMD
    from steiner_exact import solve_status

    print("\nT6  solver status guard: a time-limited incumbent is not 'Optimal'")

    class _Fake:
        pass

    def status_for(header):
        with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False) as f:
            f.write(header + "\n")
            path = f.name
        try:
            st, sol = COIN_CMD().get_status(path)
        finally:
            os.unlink(path)
        prob = _Fake()
        prob.status, prob.sol_status = st, sol
        return pulp.LpStatus[st], solve_status(prob)

    raw, guarded = status_for("Optimal - objective value 123.0")
    print(f"        proven optimum        : LpStatus={raw!r} -> {guarded!r}")
    check("a proven optimum reads as Optimal", guarded == "Optimal")

    raw, guarded = status_for("Stopped on time - objective value 123.0")
    print(f"        time limit + incumbent: LpStatus={raw!r} -> {guarded!r}")
    check("PuLP's problem status alone would have said Optimal here", raw == "Optimal",
          "this is the trap the guard exists for")
    check("the guard refuses to call it Optimal", guarded != "Optimal", f"got {guarded!r}")

    raw, guarded = status_for("Infeasible - objective value 0.0")
    print(f"        infeasible            : LpStatus={raw!r} -> {guarded!r}")
    check("an infeasible run is not Optimal", guarded != "Optimal")


def main():
    t1_star_triangle()
    t2_tiebreak()
    t2c_stage_consistency()
    t3_frozen()
    t4_generated()
    t5_percentiles()
    t6_status_guard()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All cost-accounting checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
