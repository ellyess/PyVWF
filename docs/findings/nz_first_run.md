# New Zealand — first run (the first new-climate region since the survey)

Trained 2019–2023, evaluated on held-out **2024**, EMI per-farm monthly CF,
external combined curve library, k-means++ defaults. Region config
`configs/regions/nz.toml`, source `emi-nz`. Run:
`output/validation/NZ/{train-k147, evaluate-2024-k147}`.

## Headline: the affine correction wins on every metric — NZ behaves like Denmark, not Australia

| variant | MBE | MAE | RMSE | Pearson r |
|---|---|---|---|---|
| uncorrected | −0.062 | 0.143 | 0.157 | 0.639 |
| affine k=1 fixed | −0.031 | 0.128 | 0.143 | 0.622 |
| affine k=4 fixed | +0.025 | 0.081 | 0.111 | 0.603 |
| **affine k=7 fixed** | **+0.021** | **0.0777** | **0.106** | 0.663 |
| affine k=7 season | +0.021 | 0.0789 | 0.108 | 0.644 |

12 farms, 137 farm-months in the test year.

**Best config (k=7, fixed): MAE −45% (0.143→0.078), RMSE −32%
(0.157→0.106).** This is a level-and-scale win of the same character as
Denmark (D2: level-dominated regions gain broadly), and the opposite of
Australia (near-unbiased, where the correction added farm-level noise). NZ is
the fourth region to land clearly in the "correction earns its keep" camp
(DK, DE, UK, now NZ), and the first Southern-Hemisphere one that does.

## Bias structure (the D2 diagnosis, run first)

- **ERA5 under-predicts NZ wind: uncorrected MBE −0.062** against a fleet-mean
  observed CF near 0.36. Negative bias in a high-wind, complex-terrain fleet
  (Manawatu Gorge, Cook Strait / West Wind funnelling, Te Uku ridge) is
  physically what you expect: 0.25° reanalysis smooths the orographic
  acceleration these sites are sited on. This is the same *kind* of gap the ML
  re-test flagged as globally under-represented (Tehachapi-type terrain), now
  with the sign confirmed on real data.
- **The bias is spatially structured, not uniform.** k=1 barely helps (MAE
  0.143→0.128); the jump is k=1→k=4 (0.128→0.081). A single fleet-wide factor
  leaves most of the error on the table — different farms need different
  corrections, and clustering is what captures it. Consistent with the
  us_br finding that `cluster_list=[1]` leaves 20–50% of achievable reduction
  unclaimed.
- **Correlation is near-flat** (0.64→0.66): the win is in level and scale, not
  timing. That is exactly what an affine-in-wind correction can fix and is the
  signature of a resource-magnitude bias rather than a phase error.
- **Fixed ≈ season here.** Seasonal slicing does not beat fixed (k=7: 0.0777
  vs 0.0789) — NZ's year-round westerly regime has modest seasonal amplitude,
  so there is little seasonal shape for the correction to exploit, unlike the
  trade-wind or monsoon regions.

## The k ceiling was lower than estimated — recorded so it is not re-tripped

The config first shipped `cluster_list=[1,5,10]` on the estimate of ~11 unique
coordinates. **k=10 crashed**: k-means requires `n_clusters ≤ n_samples`, and
only **8 farms reach the clusterer** in the 2019–2023 window. 13 farms − 2
post-window commissions (Harapaki 2023-11, Kaiwera Downs 2 2026) = 11 with
training obs; three more drop inside `train_set` (sparse coverage / sim-obs
merge) → 8. The sweep is now `[1,4,7]`, safely under that ceiling; k near 8
would be one-farm-per-cluster (the fake plateau). The **test** year evaluates
all 12 farms present in 2024 — the training ceiling does not limit evaluation.

Provenance note: the first (aborted) `[1,5,10]` train wrote k=1 and k=5 before
failing at k=10; reusing the run-name left a stale `k=5` in the metrics. The
committed run (`train-k147`) was produced fresh after deleting the stale
directory, so its factors are exactly `[1,4,7]`.

## Caveats carried into this result

- **Te Rere Hau is degraded** (5 turbines stopped, 2 derated late in the
  window): its observed CF understates the resource, so its farm-level
  correction partly absorbs a mechanical fault as if it were reanalysis bias.
  An exclusion-run robustness check is the obvious next probe (the AU
  far-north pattern).
- **Hub heights are hand-compiled**, three unverified (Tararua III, Mill
  Creek, Kaiwera Downs 2); `height_source` flags them in the farm table.
- **No curtailment screen.** NZ's hydro-dominated system curtails little wind
  over this window, but metered injection is net of it — a standing caveat,
  unlike BR.
- **Single seed, single test year.** The handoff's noise-floor concern applies:
  a second test year would make the fixed-vs-season near-tie conclusive rather
  than suggestive.
