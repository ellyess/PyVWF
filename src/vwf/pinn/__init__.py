"""Physics-informed bias correction for ERA5 wind inputs.

The affine correction in :mod:`vwf.correction` fits two free parameters per
cluster per time slice, directly against observed generation, and those
parameters do not transfer to regions they were not fitted on
(``docs/findings/method-ml-transfer.md``). The diagnostics in
``docs/findings/method-physics-informed-diagnostics.md`` locate the problem in the two-stage
design rather than in the choice of regressor: free per-cluster factors are
estimated noisily, depend on an arbitrary k-means partition, and collapse every
turbine-month into ~100 cluster summaries before any model sees them.

This package replaces that with a single stage. The correction is a smooth
function of local physiography, pushed through a differentiable forward
operator (wind profile, sub-daily distribution, power curve, monthly mean) and
fitted directly to observed generation. Every learned quantity is a bounded
physical parameter, so the physics rather than the network does the
extrapolating into unseen regions.
"""
