"""
Neural posterior estimation (NPE) of colliding-wind binary parameters from
short, noisy Hα photon-count time series.

Provides the forward simulator (hydrodynamics + Hα post-processing), the
detector-noise model, the prior, the spatio-temporal embedding networks, and
helpers to load data and trained posteriors. The runnable pipeline steps are
the scripts in the SBI directory.
"""
