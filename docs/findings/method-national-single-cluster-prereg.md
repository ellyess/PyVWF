# Does a single national fit lose anything against the grid's own structure?

**Date:** 2026-09-15. Registered before any run.
**Scope:** the country-level configurations BE, ES, FR, IE, IT, NL and NO.
Portugal and Sweden are excluded before any result; see the data-quality
section. Terms follow `CONTEXT.md`.

## The question

> For a country fitted against one national generation series, does giving each
> of its grid clusters its own correction beat giving the whole country one?

This began as the country-level half of `method-cluster-selection-prereg.md`
and is registered separately because it is a different question.
**A country-level configuration has no cluster count to select.**
`vwf.data.assign_country_clusters` accepts 1, or the number of clusters the
grid points already carry, and refuses everything else, because no clustering
step runs on the country path: the grid points arrive with their `cluster`
column set, and for the zonal countries it holds the bidding zones. So there is
no grid to search and no minimum to find. There are two candidates:

| BE | ES | FR | IE | IT | NL | NO | ~~PT~~ | ~~SE~~ |
|---|---|---|---|---|---|---|---|---|
| 1 or 3 | 1 or 4 | 1 or 10 | 1 or 3 | 1 or 3 | 1 or 5 | 1 or **4** | excluded | excluded |

**Corrected 2026-09-15.** This table was first written from the control-point
pool, which is built on the uniform grids. The candidate set is a property of
the **maintained** grid each configuration actually loads, and two differ:
Norway is 1 or 4, not 1 or 5, and Portugal is 1 or 2, not 1 or 3. The
fleet-weighting step empties a cluster in each and
`weight_country_grid_points.py` warns when it does. Portugal is excluded
regardless, so only Norway's entry changes a runnable candidate set. The counts
above are read by resolving each region the way `EntsoeFileSource` resolves it,
not by listing files (`AGENTS.md`).

Calling that a cluster-count selection would be a category error, and the
prediction it produced in the parent registration (C-P4, every country row
selects below 10) was retired rather than scored, because two candidates below
10 make it true by construction.

**Why it is worth asking anyway.** The standing caveat on country-level results
is that N offsets are fitted against one national series per month, so they are
under-determined and largely repair the scalar's cube-law overshoot rather than
capturing an additive spatial bias. If that is right, the grid's own structure
buys little and one national cluster is the honest model. If it is wrong, the
structure carries real spatial information and the caveat should be narrowed.
Either answer is worth having, and the run costs under 1.6 hours measured, the
Belgian anchor being 79.3 seconds for one configuration's train and
evaluate.

## Protocol

**The same one as the parent registration**, which is what makes the two
comparable: forward chaining inside the training years, the one-standard-error
rule, both metrics with the smaller-count tiebreak, then a refit on all
training years and a single untouched test year. Two candidates rather than a
grid, so the rule reduces to: **take the grid's own count only if its mean fold
score is better than one cluster's by more than one standard error of the
better mean.** Otherwise take one.

Held constant with the parent: the `fixed` time slice, `input/era5/EU_2026-09`
with the per-timestep roughness, each configuration's shipped bounding box, its
curve library, and its single test year.

**The same two conservatisms compound here**, and for the same reason: forward
chaining gives the early folds less data, and the one-standard-error rule
prefers the simpler model. Both favour one cluster. **Where a result says one
cluster is enough, that has to be said beside it.** Here it matters more than
in the parent, because "the simpler model wins" is the answer the protocol is
biased toward and also the answer the standing caveat predicts.

## Data quality, declared before the run

Six of the twenty country-level series fail a gate in
`scripts/analysis/audit_country_observations.py`, and two configurations rest
on a denominator this study should not treat as sound:

**Two configurations are excluded for the same defect and opposite causes.**
Both registers are wrong. Portugal's is wrong and repairable; Sweden's is wrong
and not repairable from anything currently held.

- **PT, excluded: the register is wrong and can be repaired.** It is flat at
  4,486 MW for 2015 through 2019 while the Global Wind Power Tracker has the
  fleet growing from 4,355 to 4,722 MW. GWPT disagrees by at most 5% and moves
  where the register does not, repairing it leaves every year's peak capacity
  factor between 0.95 and 0.98, and it changes 52 of 84 training months by more
  than 0.01. Portugal returns once its series is repaired by the route Ireland
  took.
- **SE, excluded: the register is derived from the generation it divides.**
  It is flat at 8,354 MW for 2015 through 2019. **Corrected 2026-09-15:** this
  was first recorded as a case where the register is wrong and the Global Wind
  Power Tracker is not the fix, on the grounds that repairing from GWPT puts
  Sweden's 2015 national mean capacity factor at 0.4481 while the current
  0.2267 looks like a real fleet. **The current value is not evidence**: the
  four Swedish bidding zones each carry a capacity frozen across all five
  training years, each series peaks at exactly 0.900, and the four sum to
  8,354 MW exactly. 0.900 is the signature of the ENTSO-E fetcher's fallback,
  `estimated_cap = gen.max() / 0.9`, so the denominator is back-derived from
  the numerator and the series is constructed to look like a national fleet
  whether or not it is one. Which register is right for Sweden is open, and
  the current one has the weaker claim, not the stronger. **Sweden needs a
  real installed-capacity register**, logged as an open item rather than
  searched for now, and returns when it has one.

**Neither exclusion is lifted by seeing a number.** Both are fixed here, before
any result, which is what separates an exclusion from a convenience. Sweden
passes every gate the auditor currently applies, and was found only by testing
the proposed frozen-register rule, which is recorded so the exclusion does not
read as hindsight.
- **NL.** The documented coverage defect: peak capacity factor never exceeds
  0.570 and the annual mean ranges 3.4x across the record. NL is reported and
  its result is read against that, which is the existing treatment.
- BE, IE and NO each carry a small number of hours above 1 against an annual
  register that cannot track within-year additions. They are kept; the reach is
  one hour in 61,364 for BE, one in 87,353 for IE and 31 in 61,286 for NO.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **N-G1** | The protocol completes for all seven included configurations under both metrics. | |
| **N-G2** | The grid's own count is selected over one cluster in at least three of the seven. Below that, the structure is not earning its place. | |
| **N-G3** | Where the grid's own count is selected, it beats one cluster on the test year by more than 0.002 MAE. A selection that does not survive to the test year is reported as not surviving. | |

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| N-P1 | **N-G2 fails.** One cluster is selected in five or more of the seven, because the national series cannot identify more and because both conservatisms push that way. | |
| N-P2 | France is the most likely exception, having ten clusters over the largest area and the most spatial spread to carry. | |
| N-P3 | The Netherlands selects one cluster, and its result should not be read as evidence either way, given its coverage defect. | |
| N-P4 | RMSE and MAE agree on the choice in at least five of the seven, the choice being binary and the two metrics rarely differing on a binary. | |

## Committed in advance

- Portugal's and Sweden's exclusions are declared above, before any result,
  and are lifted only by fixing their registers, not by seeing their numbers.
- All seven included configurations are reported, including those where one
  cluster wins, which N-P1 expects to be most of them.
- The protocol's bias toward the simpler model is stated beside every result
  that selects one cluster.
- If N-G2 fails, that is the finding, and it narrows to a statement about what
  a single national series can identify rather than about clustering.
