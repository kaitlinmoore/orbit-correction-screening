"""Learned correction of SGP4 predictions for LEO conjunction screening.

Modules:
    types         contracts between components (trajectory, residual schema, correction interface)
    frames        TEME -> GCRF conversion and time-scale helpers
    catalog       element-set and SATCAT loading, LEO and quality filters
    spaceweather  CelesTrak space-weather file: parsing, leak-free lookups, Orekit-compatible writer
    splits        object and time splits for training and held-out evaluation
"""

__version__ = "0.1.0"
