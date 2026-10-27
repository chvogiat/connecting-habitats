"""
smoke_test.py
=============
Run this first on any new machine. It checks that a student can actually run the
module, not merely that the packages are installed:

  1. records the platform and package versions;
  2. starts the CBC executable and solves a tiny integer program with it;
  3. runs the starter (Kou heuristic), one exact Steiner solve and one survivable solve
     on the frozen instance for the eastern box turtle, and compares the three costs
     with the published values.

    python smoke_test.py

Exits non-zero on any failure. A solver that is listed as available but cannot start
(for example PuLP's x86_64 CBC on Apple silicon without Rosetta 2) fails in step 2.
"""
import platform
import sys
import time

import networkx as nx
import matplotlib
import pulp

from corridor_starter import (SPECIES_PROFILES, assign_weights, kou_steiner,
                              load_graph, total_weight)
from steiner_exact import cbc_path, exact_steiner
from corridor_redundant import redundant_corridor

PUBLISHED = {"kou": 660_270.6, "steiner": 579_635.5, "survivable": 881_511.3}
REL_TOL = 1e-6
failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def main():
    print("1. Environment")
    print(f"     Python {platform.python_version()} on {platform.system()} "
          f"{platform.release()} ({platform.machine()})")
    print(f"     networkx {nx.__version__}, PuLP {pulp.__version__}, "
          f"matplotlib {matplotlib.__version__}")

    print("2. Solver startup")
    try:
        path = cbc_path()
    except RuntimeError as err:
        check("a CBC executable starts on this machine", False, str(err))
        return 1
    print(f"     CBC: {path}")
    # Figure 2(a) of the article: the exact optimum is the three-spoke star, cost 3.
    G = nx.Graph()
    for t in "abc":
        G.add_edge("s", t, weight=1.0)
    for u, v in [("a", "b"), ("a", "c"), ("b", "c")]:
        G.add_edge(u, v, weight=1.9)
    _, opt, status, _ = exact_steiner(G, ["a", "b", "c"], time_limit=60)
    check("tiny integer program solves to proven optimality",
          status == "Optimal" and abs(opt - 3.0) < 1e-6, f"{status}, {opt}")

    print("3. Frozen instance, eastern box turtle")
    G, terminals, _, _ = load_graph()
    assign_weights(G, SPECIES_PROFILES["eastern_box_turtle"])
    results = {}
    t0 = time.time()
    results["kou"] = total_weight(kou_steiner(G, terminals))
    _, results["steiner"], st1, sec1 = exact_steiner(G, terminals, time_limit=900)
    _, results["survivable"], st2, sec2 = redundant_corridor(G, terminals, time_limit=900)
    print(f"     Steiner {st1} in {sec1:.1f} s; survivable {st2} in {sec2:.1f} s")
    check("both exact solves proven optimal", st1 == "Optimal" and st2 == "Optimal")
    for key, ref in PUBLISHED.items():
        got = results[key]
        check(f"{key} cost matches the article ({ref:,.1f})",
              abs(got - ref) <= REL_TOL * ref + 0.05, f"got {got:,.2f}")
    print(f"     total {time.time() - t0:.1f} s")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s).")
        return 1
    print("Smoke test passed: this machine can run the module.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
