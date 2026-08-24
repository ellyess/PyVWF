# E10: profile curvature. The physics works, the model does not want it.

Registered in addendum 9 before implementing, with three predictions. Two
confirmed, the decisive one failed, and the term is **not adopted**. As with the
wake term, what it establishes is worth more than the accuracy it did not buy.

## What was built

The power law and the log law disagree about curvature in `ln z`, and the
disagreement grows with distance from the 10-100 m band the exponent is measured
over. Extrapolated, a power law gives about 1% LESS wind than the log profile
that generated it at 30 m and about 1% MORE at 150 m, 2-3% at rough sites.

The measured exponent inverts to a roughness in closed form,

    ln z0 = ln(10) * (r - 2) / (r - 1),    r = w100/w10 = 10**shear

so the log law can be applied with an HOURLY roughness: the right curvature and
the stability response together, where before it was one or the other. Verified
exact -- a shear of 0.145 returns z0 = 0.03000 m, and the profile matches the log
law to 0.000% at every height where the power law was off by up to 1.2%. The
learned correction becomes an offset to `ln z0` rather than to the exponent.

## The three predictions

**P1 -- Germany improves most: CONFIRMED.**

| holdout | power | shear-log | change |
|---|---|---|---|
| **DE** | 0.0678 | **0.0671** | **−0.0007** |
| BR | 0.1030 | 0.1038 | +0.0008 |
| UK | 0.1363 | 0.1374 | +0.0011 |
| US | 0.1001 | 0.1013 | +0.0012 |
| DK | 0.0951 | 0.0968 | +0.0017 |

Germany is the only region that improves and by the largest margin, which is
where the physics says curvature should bite: the tallest fleet, 40 to 149 m,
and 91% of the 120-200 m residual bin.

**P3 -- the tall-turbine residual falls: CONFIRMED.** The 120-200 m bin drops
from **+0.0361 to +0.0136**, a 62% reduction. But the residual span across ALL
height bins RISES, 0.0455 to 0.0539: the error moved rather than left.

**P2 -- mean transfer skill holds: FAILED.** 0.3378 to 0.3289, a fall of 0.0089
against a 0.005 tolerance.

## Why it fails, measured rather than guessed

The comparison changed two things at once. Pooled over 14,998 units in nine
regions, the median span each head can impose on the profile ratio:

| form | bounds | median span |
|---|---|---|
| power (shear-exponent offset) | −0.15 to +0.25 | **13.30%** |
| shear-log (ln z0 offset) | ±2.5 | 4.24% |

So the new form is also less than a third as expressive. Widening its bounds
does not fix that: the span **saturates at 7.32%** and cannot reach 13.30% at
any bound, because z0 hits its physical clamp of 1e-6 to 2 m.

That is the finding, and it is sharper than the headline:

> **The power law's extra leverage consists of profile shapes no physical
> roughness can produce.** A shear-exponent offset of −0.15 to +0.25 reaches
> profiles that no roughness in the physical range generates. The model has been
> using that unphysical freedom, and it is worth about 0.001 RMSE in transfer.

So this is not "curvature costs accuracy". It is "confining the profile to
physically realisable shapes costs accuracy", which says the profile term has
been absorbing error that does not belong to it. The term is not adopted, and it
is deliberately NOT promoted by widening a bound past the physical range, since
that freedom is precisely what the result indicts.

## The pattern across E9 and E10

Two physics terms, both derived from the residual anatomy, both behave the same
way:

| | targeted defect | did it fix it? | transfer |
|---|---|---|---|
| E9 wake | dense-fleet over-prediction | **yes**, span −61% | **worse**, 5/5 |
| E10 curvature | tall-turbine over-prediction | **yes**, bin −62% | **worse**, 5/5 |

Both removed the systematic they aimed at, both **redistributed error onto other
axes** rather than removing it, and both cost transfer. That is not two
coincidences. D6 already showed the efficiency and speed-up terms trade off
along a nearly flat direction; a model with that much compensating freedom
responds to a new constrained term by re-absorbing the error elsewhere, and what
it gives up is freedom the fit was using.

The implication for what comes next is a change of direction. The remaining
leads in `method-physics-informed-evaluation.md` were all "add a physics term",
ranked by residual size. Two have now been tried and both failed the same way.
The evidence points instead at **identifying the terms already present** --
D6's flat direction between efficiency and speed-up, and the unphysical profile
freedom found here -- before adding a fourth term for them to trade against.

A concrete first step, not taken here: constrain the existing terms jointly
rather than one at a time, and check whether transfer improves when the model
has LESS freedom rather than more. E9 and E10 each removed freedom in one place
and lost; whether removing it in several places at once behaves differently is
an open and cheap question.

## Disposition

Implemented, covered by six tests, **off by default**, and correct for anyone
working a tall fleet: Germany gains from it and Germany is the only region here
whose hub heights make the question meaningful. `profile="power"` reproduces
every existing result.
