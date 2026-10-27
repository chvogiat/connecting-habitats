# Connecting Habitats: minimal replication code

This folder has the code and data needed to reproduce every computed table, figure and
number in *Connecting Habitats: From Optimal Trees to Survivable Networks*, plus four optional projects and the
browser explorer. Everything runs offline with free software.

## Setup

    conda env create -f environment-current.yml
    conda activate corridor
    python smoke_test.py

`smoke_test.py` starts the CBC solver, solves a tiny integer program, and reproduces the
eastern box turtle's heuristic, Steiner and survivable costs. Run it first on any new
machine. If it reports that no CBC executable starts, install `coin-or-cbc` from conda-forge
(the environment file already does this) or set `CORRIDOR_CBC` to a CBC binary.

## Reproducing the article

Run each command from this folder. Each one writes its outputs next to the scripts.

| Article object | Command | Expected output |
|---|---|---|
| Table 2 (resistance multipliers) | — (values in `SPECIES_PROFILES` in `corridor_starter.py`) | 8 road classes; 115/76/71/68/67/15/14/12 segments in `edges.csv` |
| §2.4 hand examples, Table 3, and the four costs | `python check_cost.py` | all checks pass |
| Table 4 (heuristic vs exact) | `python steiner_exact.py` | 182,780 / 178,817 / 159,461 for deer and the other two rows, all `Optimal`; run times depend on the machine |
| Table 5, Figure 5 (survivability) | `python corridor_redundant.py` | premiums +41.5% / +52.1% / +52.1%; writes `fig8_redundancy.png` |
| §3.2 LP relaxation claim | `python lp_relaxation_check.py` | LP bound equals the integer optimum for all three species |
| §3.2 solver records | `python solver_records.py` | six solves, all `Optimal`, zero branching nodes |
| Figure 1 (study area) | `python make_fig1.py` | `fig1_studyarea.png` |
| Figures 3 and 4 (heuristic vs exact; three species) | `python make_fig_species.py` | `fig4_exact_vs_kou.png`, `fig2_species.png` |
| Table 6 (generated family) | `python generate_instance.py --sweep 20` | 120 exact solves, all `Optimal`, a few minutes; **overwrites** `robustness_sweep.json` |
| §5 road networks | `python real_data_feasibility.py net_extracts/*` | all five `feasible as placed: False`; 13.8–55.0% of edges are bridges (the `README.md` in that folder is skipped) |
The figure filenames date from an earlier draft: `fig4_exact_vs_kou.png` is Figure 3,
`fig2_species.png` is Figure 4, and `fig8_redundancy.png` is Figure 5. Figure 2 is drawn in
LaTeX and has no script.

`check_cost.py` also regenerates the 60 sweep instances from the seeds stored in
`robustness_sweep.json` and checks their heuristic costs, so you can confirm Table 6's
instances without rerunning all 120 exact solves.

## Browser explorer

Open `corridor_dashboard.html` in any browser. It is self-contained and needs no network
access or installation. It runs the Kou heuristic interactively on the teaching instance,
not the exact model.

## Files

| File | Role |
|---|---|
| `nodes.csv`, `edges.csv` | The teaching instance: 224 nodes, 438 edges, 8 habitat patches |
| `corridor_starter.py` | Session 1: graph loading, species profiles, Kou–Markowsky–Berman heuristic |
| `steiner_exact.py` | Session 2: exact multi-commodity flow model and the CBC solver setup |
| `corridor_redundant.py` | Session 3: terminal 2-edge-connected survivable model |
| `smoke_test.py` | Installation check |
| `check_cost.py` | Checks for the hand examples, the four costs and the sweep instances |
| `lp_relaxation_check.py`, `solver_records.py` | Evidence behind the §3.2 solver claims |
| `make_fig1.py`, `make_fig_species.py` | Figures 1, 3 and 4 |
| `generate_instance.py`, `robustness_sweep.json` | §5 generated family and its recorded seeds |
| `real_data_feasibility.py`, `net_extracts/` | §5 road networks (five archived OpenStreetMap extracts) |
| `corridor_dashboard.html` | Browser explorer |
| `environment-current.yml` | Tested software environment |

Read `net_extracts/README.md` before using the road networks: their edge lengths were
repaired after download, and it explains the resulting limitations.

## Optional projects

These extensions are independent of the three core sessions and use the same environment
and teaching instance. Run commands from this folder.

| Project | Command | Teaching question |
|---|---|---|
| Terminal quotas | `python corridor_budget.py` | What changes when only the root and a chosen number of other patches must connect? |
| Uncertain resistance | `python corridor_robust.py` | Which routes persist under independent lognormal shocks to road-class multipliers? |
| Commitments | `python corridor_consensus_fix.py` | What happens when frequently selected segments are committed at zero marginal cost in subsequent rounds? |
| Profile sensitivity | `python profile_sweep.py` | How does the survivability premium depend on relative road-class weights? |

The quota script sweeps the required patch count for all three species and writes
`budget_frontier.json` and `fig5_budget_frontier.png`. This is a rooted quota model,
not a fixed-budget or prize-collecting model. Designs need not be nested as the quota grows.
To run one species, use `python corridor_budget.py eastern_box_turtle`.

The uncertainty script defaults to 400 Kou heuristic runs for the turtle with lognormal
shocks of median 1 and sigma 0.35, and writes `fig6_consensus_corridor.png`.
Selection frequencies describe this procedure, not ecological correctness; frequently
selected segments need not form a connected corridor. For a shorter run, use
`python corridor_robust.py 20`; `--exact` switches to exact Steiner solves.

The commitment script defaults to the turtle, 400 draws per round, and a 90% selection
threshold. It writes `consensus_fixing_results.json` and, when segments are committed,
`fig9_consensus_fixing.png`. Committed segments can remain unused, and zero marginal
cost represents a sunk-cost convention rather than movement resistance. Use `--all`
for all species or `--greedy` for the alternative completion procedure, which writes
`consensus_greedy_results.json` and frontier figures.

The profile script varies both road aversion and the ordering of class multipliers and
writes `profile_sweep.json`. Uniformly scaling all weights changes neither the optimal
design nor the survivability premium. This project requires many exact solves.

Stored development results are included as `budget_frontier.json`,
`consensus_fixing_results.json`, `consensus_greedy_results.json`, and `profile_sweep.json`.
These are archived results, not evidence of a fresh rerun in the current environment.
The commands above overwrite or update their corresponding files; preserve copies before
rerunning if the original records are needed. To inspect results without repeating the
solves, read the JSON files or regenerate the quota and profile figures with:

    python corridor_budget.py plot
    python -c "from profile_sweep import make_figures; make_figures()"

## Not included

The scripts that download and repair OpenStreetMap extracts, the dashboard builder,
the historical environment record, and other outputs from the full development package
are not included. None is needed to reproduce the article or run the four optional
projects. Some docstrings and `net_extracts/README.md` mention those files.

## License

The code is released under the MIT License; see `LICENSE`. The road-network extracts in
`net_extracts/` are not covered by it: map data © OpenStreetMap contributors, available
under the Open Database License (ODbL).
