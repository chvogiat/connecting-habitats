r"""
Exact Steiner tree via the directed multi-commodity flow formulation,
benchmarked against the Kou heuristic and the independent-links baseline.

Formulation (rooted at r in T):
  Bidirect every edge {i,j} into arcs (i,j) and (j,i), each with the edge weight.
  Variables:
      x_a in {0,1}            arc a is used
      f^k_a >= 0              flow of commodity k on arc a   (k in T\{r})
  min  sum_a w_a x_a
  s.t. flow conservation: for each commodity k and node i,
          sum_{a out of i} f^k_a - sum_{a into i} f^k_a = +1 if i=r, -1 if i=k, else 0
       coupling:  f^k_a <= x_a   for all arcs a, all commodities k
  Each commodity ships one unit from r to its terminal k; coupling forces the
  union of flow-carrying arcs to be selected. The optimal selection is an
  arborescence rooted at r whose undirected support is a minimum Steiner tree.
"""
import os
import shutil
import subprocess
import time
import pulp
from corridor_starter import (load_graph, assign_weights, kou_steiner,
                              total_weight, SPECIES_PROFILES)
import networkx as nx


# The settings every reported exact run uses. Stated explicitly rather than left
# to whatever the bundled CBC happens to default to, so that one documented
# configuration covers this script, corridor_redundant.py and solver_records.py.
GAP_REL = 0.0
THREADS = 1


_CBC_PATH = None


def _runs(path):
    """True if the executable at ``path`` starts on this machine."""
    try:
        subprocess.run([path, "-quit"], capture_output=True, timeout=30, check=False)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def cbc_path():
    """Locate a CBC executable that actually runs here, and remember it.

    Order of preference:
      1. the path in the environment variable CORRIDOR_CBC, if set;
      2. the binary bundled with PuLP (the one behind every reported number);
      3. a ``cbc`` on PATH (for example ``conda install -c conda-forge coin-or-cbc``).
    PuLP's bundled binary is x86_64 on macOS, so on an Apple-silicon machine
    without Rosetta 2 it exists but fails with "Bad CPU type in executable".
    Checking that a file exists does not catch that; starting it does.
    """
    global _CBC_PATH
    if _CBC_PATH:
        return _CBC_PATH
    candidates = [os.environ.get("CORRIDOR_CBC")]
    try:
        candidates.append(pulp.PULP_CBC_CMD().path)
    except Exception:
        pass
    candidates.append(shutil.which("cbc"))
    for path in candidates:
        if path and os.path.exists(path) and _runs(path):
            _CBC_PATH = path
            return path
    raise RuntimeError(
        "No working CBC executable found. PuLP's bundled CBC does not run on this "
        "machine; install one (e.g. `conda install -c conda-forge coin-or-cbc`) or "
        "set CORRIDOR_CBC to its path. See README.md and smoke_test.py.")


def cbc_solver(**kwargs):
    """The CBC command every exact solve in this package uses (see cbc_path)."""
    return pulp.COIN_CMD(path=cbc_path(), **kwargs)


def solve_status(prob):
    """Return a status string in which ``"Optimal"`` means *proven* optimal.

    PuLP's problem status alone cannot carry that meaning. Its CBC reader
    classifies a solution file by the first word of the header, and then adds
    this rule (``pulp/apis/coin_api.py``, PuLP 2.7.0)::

        if status == LpStatusNotSolved and len(statusstrs) >= 5:
            if statusstrs[4] == "objective":
                status = LpStatusOptimal
                sol_status = LpSolutionIntegerFeasible

    So a run that hits its time limit holding an incumbent writes
    ``"Stopped on time - objective value ..."`` and comes back as
    ``LpStatus[prob.status] == "Optimal"``, exactly like a proven optimum. Only
    the *solution* status distinguishes them: ``LpSolutionOptimal`` for a proven
    optimum, ``LpSolutionIntegerFeasible`` for the time-limited incumbent. A
    time-limited run with no incumbent does report ``"Not Solved"``, which is why
    the difference is easy to miss -- the dangerous case is the one that looks
    like success.

    Every caller in this package compares against the string ``"Optimal"``, so
    narrowing the string here repairs all of those checks at once.
    """
    problem_status = pulp.LpStatus[prob.status]
    if problem_status != "Optimal":
        return problem_status
    sol_status = getattr(prob, "sol_status", None)
    if sol_status == pulp.LpSolutionOptimal:
        return "Optimal"
    if sol_status == pulp.LpSolutionIntegerFeasible:
        # Feasible incumbent, no matching lower bound: an upper bound only.
        return "Feasible (time limit; not proven optimal)"
    return f"Optimal (unconfirmed: sol_status={sol_status})"


def exact_steiner(G, terminals, time_limit=120, msg=False):
    # Timed from here: the reported figure is wall clock INCLUDING model
    # construction, which is the convention stated in the article. Building the
    # multi-commodity model is a real part of the cost of answering the question,
    # and on these instances it is not negligible beside the solve.
    t0 = time.time()
    r = terminals[0]
    commodities = terminals[1:]

    # Directed arcs with weights.
    arcs = []
    w = {}
    for i, j, d in G.edges(data=True):
        arcs.append((i, j)); w[(i, j)] = d["weight"]
        arcs.append((j, i)); w[(j, i)] = d["weight"]

    out_arcs = {n: [] for n in G.nodes()}
    in_arcs = {n: [] for n in G.nodes()}
    for (i, j) in arcs:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    prob = pulp.LpProblem("steiner_mcf", pulp.LpMinimize)
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

    solver = cbc_solver(msg=msg, timeLimit=time_limit,
                        threads=THREADS, gapRel=GAP_REL)
    prob.solve(solver)
    elapsed = time.time() - t0

    status = solve_status(prob)
    H = nx.Graph()
    for (i, j) in arcs:
        if x[(i, j)].value() and x[(i, j)].value() > 0.5:
            H.add_edge(i, j, weight=w[(i, j)])
    return H, pulp.value(prob.objective), status, elapsed


if __name__ == "__main__":
    G, terminals, pos, names = load_graph()
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(terminals)} terminals\n")
    print(f"{'Species':28s} {'Baseline':>12s} {'Kou':>12s} {'Exact':>12s} "
          f"{'Kou/Exact':>10s} {'sec':>7s}")
    for key, profile in SPECIES_PROFILES.items():
        assign_weights(G, profile)
        # independent-links baseline
        closure = nx.Graph()
        for i, s in enumerate(terminals):
            dist = nx.single_source_dijkstra_path_length(G, s, weight="weight")
            for t in terminals[i + 1:]:
                if t in dist:
                    closure.add_edge(s, t, weight=dist[t])
        base = sum(d["weight"] for _, _, d in
                   nx.minimum_spanning_tree(closure, weight="weight").edges(data=True))
        kou = total_weight(kou_steiner(G, terminals))
        _, opt, status, sec = exact_steiner(G, terminals)
        print(f"{profile['description'][:28]:28s} {base:12,.0f} {kou:12,.0f} "
              f"{opt:12,.0f} {kou/opt:10.4f} {sec:7.1f}  [{status}]")
