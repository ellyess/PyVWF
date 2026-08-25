# What predicts whether a new region will gain? Two answers, both wrong.

A practitioner's first question about a transferable correction is whether it
will help *their* region. This records two candidate answers, the evidence for
each, and the evidence that neither survives. It is a negative result, and the
useful part is that both were offered with numbers behind them before the data
that refuted them existed.

## The decomposition the question is usually framed in

A region's uncorrected mean squared error splits exactly in two:

    MSE  =  MBE^2  +  Var(error)
            level     shape

**Level** is a constant offset: the reanalysis has the pattern right and the
average wrong, so a wind-speed rescaling fixes it precisely, because moving all
the wind moves all the output. **Shape** is everything else, and a single scalar
cannot touch it.

`method-generalisation.md` established for the CLUSTER-LEVEL AFFINE correction
that it "earns its keep where the reanalysis bias is level-dominated", with
Denmark (level-dominated, gains on every metric) against Australia
(near-unbiased, absolute skill gets worse).

## The nine regions, decomposed

| region | uncorr RMSE | MBE | level share | shape share | zero-shot skill |
|---|---|---|---|---|---|
| DK | 0.1464 | +0.109 | **55.6%** | 44.4% | +0.544 |
| DE | 0.0860 | +0.042 | 24.3% | 75.7% | +0.156 |
| NZ | 0.1567 | −0.062 | 15.6% | 84.4% | **+0.707** |
| BR | 0.1386 | −0.046 | 11.0% | 89.0% | +0.413 |
| UK | 0.1451 | +0.036 | 6.2% | 93.8% | +0.031 |
| CL | 0.1225 | −0.027 | 4.8% | 95.2% | −0.091 |
| US | 0.1098 | +0.022 | 4.0% | 96.0% | +0.046 |
| AU-NEM | 0.1153 | +0.009 | **0.6%** | 99.4% | **+0.534** |
| AR | 0.1512 | +0.010 | 0.4% | 99.6% | +0.268 |

Skill is configuration A, zero-shot, each region held out.

## Candidate 1: physiographic coverage. Refuted.

D5 measured how much of a held-out region's physiography the training regions
span, and on the original five it looked convincing: Denmark 93% coverage and
+0.54 skill, the United States 50% and +0.05. The framing entered
`method-physics-informed-diagnostics.md` and the evaluation as though settled.

It does not survive nine regions. **New Zealand has the lowest coverage of all
(33.3%) and the highest skill (+0.707); Argentina has among the highest (91.5%)
and less than half of it.** Pooled: pearson **+0.224, p = 0.56**.

## Candidate 2: level dominance. Also refuted.

Offered as the replacement when coverage failed, on the strength of |MBE|
correlating +0.562 with skill. Tested properly, neither form holds:

| predictor | pearson | p |
|---|---|---|
| physiographic coverage | +0.224 | 0.56 |
| uncorrected RMSE | +0.226 | 0.56 |
| absolute uncorrected MBE | +0.562 | 0.115 |
| **level share (MBE^2 / MSE)** | **+0.377** | **0.32** |

**Australia refutes it outright**: 0.6% of its error is level, so there is
essentially no offset to remove, and it still gains +0.534, third best of nine.

## Why the older finding may not carry, and what is untested

There is a structural reason level-dominance might hold for the affine model and
not this one, and it is worth stating as a hypothesis rather than a conclusion.
The affine correction fits one scalar per cluster, so it can only move levels.
This model emits a correction **per site** from local physiography, so in
principle it can also repair error that is spatially structured -- the right
fleet mean with the wrong distribution across it. Australia gaining with no
level bias to remove is what that would look like.

That is a hypothesis fitted to nine points after the fact. Testing it means
decomposing each region's error into a fleet-mean part and a between-site part
and asking which the correction removes. Not done here.

## The honest state of the question

**No predictor is established.** Nine regions is too few, both candidates fail
significance, and each was proposed before the data that refuted it. What can be
said is what was measured directly: the correction improved 5 of 5 original
regions and 4 of 4 fresh ones, and harmed none of the nine under configuration D
and one of nine (Chile, −4.4%) under configuration A.

Until a predictor is established, the honest advice for a new region is to run
it and look, not to forecast from a summary statistic. Two summary statistics
have now been offered and both were wrong.
