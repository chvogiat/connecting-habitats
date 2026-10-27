# Frozen OpenStreetMap extracts

Five drivable road networks, downloaded on 2026-09-14 with `acquire_real_data.py`
(osmnx 1.9.4) and frozen here so that Section 5's real-network study reproduces without
network access. **The `length_m` column was corrected on 2026-09-16; see below.** OpenStreetMap changes continuously; re-downloading these places will not
give these files back.

| Directory | Place query |
|---|---|
| `monticello_illinois_usa` | Monticello, Illinois, USA |
| `mahomet_illinois_usa` | Mahomet, Illinois, USA |
| `ashland_oregon_usa` | Ashland, Oregon, USA |
| `ely_cambridgeshire_england` | Ely, Cambridgeshire, England |
| `piatt_county_illinois_usa` | Piatt County, Illinois, USA |

## Edge lengths: corrected 2026-09-16 — read this before using the data

The `length_m` column in the files as first downloaded was **invalid**, and has been
replaced. The acquisition script called `ox.distance.add_edge_lengths()` *after*
`ox.project_graph()`. That function reads each node's `(x, y)` as `(lon, lat)` in degrees;
on a projected graph it was therefore reading UTM eastings and northings as degrees. Every
valid OSM length was overwritten with a meaningless number — the median came out between
3.8 and 9.3 million metres, for street segments tens to hundreds of metres long.

The corrupt values could not be inverted: they are a deterministic function of the two
endpoint coordinates alone, so they carried no information about the original road
geometry, and the function is not monotone in true distance, so the error did not cancel
out of ratios (correlation with true endpoint separation ran 0.18–0.48).

**What was done.** `repair_extract_lengths.py` recomputed each length as an approximate
endpoint distance: the distance between the edge's two endpoints in a local tangent plane to
the WGS84 ellipsoid. This is not an exact ellipsoidal geodesic, and its accuracy has not been
validated over the full range of these edges, which reach 11,990.9 m. The endpoint latitudes
and longitudes came from the *unprojected* graph and were never affected. The original files
are not included in this package.

**The model change this entails.** OSMnx measures length along an edge's true road
geometry; these repaired lengths approximate the distance between the intersections at its
ends, which ignores the road's curvature between them. An edge's along-road length is always
at least its endpoint distance and can exceed it substantially on a winding alignment, but
the original geometry was not archived, so the size of that error — and therefore its effect
on any ratio of optimized costs — is unquantified. Read these as real road *topologies*
carrying artificial edge weights. Section 5 of the article states this, and every number
derived from these files inherits it.

**Two limitations the repair cannot address.** Where the original download had several
parallel alignments between the same pair of nodes, the script kept the shortest, choosing
with the corrupt lengths; which alignment survived is arbitrary, and so is the `highway`
class it carries — which sets that edge's resistance multiplier, and hence its weight. And collapsing the road
multigraph to a simple graph discards genuinely distinct parallel routes, which removes real
redundancy. Reacquiring with the corrected `acquire_real_data.py` is the only fix for
either, and is preferable to this repair whenever OSMnx and network access are available.

## Placeholder patches

The `is_habitat_patch` column holds the acquisition script's **placeholder** farthest-point
samples, not protected-area data. See the caveat paragraph in the article.

Reproduce the study from the package folder with:

    python real_data_feasibility.py --json real_data_results.json net_extracts/*

The acquisition script `acquire_real_data.py` and the repair and audit script
`repair_extract_lengths.py` are part of the full development package and are not included
here; they are not needed to reproduce the article.

Map data © OpenStreetMap contributors, available under the Open Database License (ODbL).

## Checks for a new road network

Compute edge lengths before projecting the graph, then check that their
median is plausible for street segments. Collapsing a road multigraph to a simple graph discards
real parallel routes. Treat terminals placed by a farthest-point rule as placeholders,
and run the bridge test of main Section 4 before solving the survivable model. The bridge
test depends only on topology and terminal placement, not on the repaired lengths.
