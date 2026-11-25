"""Learned correction of SGP4 predictions for LEO conjunction screening.

Modules:
    types         contracts between components (trajectory, residual schema, correction interface)
    frames        TEME -> GCRF conversion and time-scale helpers
    catalog       element-set and SATCAT loading, LEO and quality filters
    spaceweather  CelesTrak space-weather file: parsing, leak-free lookups, Orekit-compatible writer
    splits        object and time splits for training and held-out evaluation
    cheap_prop    SGP4 wrapper: element sets to GCRF trajectories, error status, 120 km floor
    truth_prop    truth propagator: Orekit numerical integration (JVM only in worker processes)
    residuals     curvilinear RTN residual and its inverse, residual-row builder, drop rule
    screening     conjunction screening engine, classical filters and screening metrics

Importing any of them never starts a JVM.
"""

__version__ = "0.1.0"
