You are grading one review of a proposal. You do not see the proposal. Below
is a description of one specific problem, then the review. Do not open any
file; judge from the text here alone.

Question: does the review identify this problem as something wrong or missing
in the proposal? Count it only if the review states the problem's mechanism,
in its own words, as a fault of the proposal. A review that mentions the same
code or setting without saying it is at fault does not count. A review that
credits the proposal with handling it does not count.

Answer in exactly two lines:
FOUND: YES or FOUND: NO
QUOTE: the shortest passage from the review that states the problem, copied
character for character; or NONE

## The problem

The audit reads `cluster_list` at the top level of `run_manifest.json`, but
the harness writes it under the `correction` block
(`manifest["correction"]["cluster_list"]`). Every run therefore appears to
declare nothing, is skipped as predating the field, and nothing is flagged:
a clean answer from a check that never ran. Running the audit first on runs
already known to be contaminated (a known positive) would have exposed it.
A review that names either the wrong key location or the missing known-positive
check on the contaminated runs identifies this defect.

## The review

**Verdict:** REJECT, as designed. The audit would report "0 flagged" whatever is actually on disk, and that result is what lets the gate pass.

**Blockers**

1. **The audit reads a key that is not there.** `build_manifest` puts the list under `manifest["correction"]["cluster_list"]`, not at the top level (`/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/b0097fba/src/vwf/harness/provenance.py:284-288`). `manifest.get("cluster_list")` returns None for every harness run, so all of the roughly 220 runs are skipped, none is flagged, and the gate passes with nothing checked. Fix: read `manifest["correction"]`. Make "skipped" a fail state, not a pass: gate on skipped count = 0, or treat skipped runs as unaudited and re-run them.

2. **Overwrites are invisible to this check.** `run_train` writes `factors_{time_res}_{num_clu}.csv` and `train_turb_info_{num_clu}.csv` into a shared directory and then rewrites the manifest last (`/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/b0097fba/src/vwf/harness/driver.py:248-291`). So the manifest lists only the last writer's counts. When two configurations share a cluster count, the second silently replaces the first's file. In the known case the onshore and offshore grids share 1, 10, 25, 50 and 100 (`docs/findings/method-cluster-selection-prereg.md:169-172`). Those rows are contaminated, and the audit cannot see them.
   - In any other study whose varied dimension is roughness, curve library, fleet `mode` or years, both configurations share one `cluster_list`. A collision there leaves no undeclared file at all.
   - The "expected effect" assumes the very thing the audit should test.
   - Fix: detect collisions directly:
     - (a) list the `(code, run_name)` pairs each study driver generates and look for duplicates across configurations;
     - (b) compare each factors file's mtime with the manifest's `created_utc`;
     - (c) compare the factors files with the manifest's `fit_diagnostics` keys, which give the combinations actually fitted, including `time_slices`. The proposal checks counts only.

3. **No check that the audit can fire.** Run it on the known DK and UK `train-fold-<year>` directories first and show it flags them. Then run it on a clean directory and show it does not. Write that requirement into the gate.

**Should fix**
- Evaluate directories collide the same way (`evaluate-{year}-{run_name}`, `driver.py:343`), so `metrics.csv` and `cor_cf_*` get overwritten too. Include them.
- Name the contaminated rows of the cluster-selection study, with their training years and single test year per fold. "Only the contaminated rows" needs that list, and from blocker 2 it is larger than the non-shared counts.
- `docs/CONTEXT.md` does not exist in this checkout, so I could not check terms against it.

**What would change my verdict:** fix the manifest path, make skipped runs fail the gate, and add a collision check that can see overwrites (path-key enumeration plus mtime). Show it flags the known DK and UK directories and stays negative on a clean one before it is run on the rest of `output/`.
