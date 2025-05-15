"""
PyCCT: Python implementation of the Cauchy Combination Test

This package provides an implementation of the Cauchy Combination Test (CCT),
a powerful p-value combination method using the Cauchy distribution.

Reference:
Liu, Y., & Xie, J. (2020). Cauchy combination test: a powerful test
with analytic p-value calculation under arbitrary dependency structures.
Journal of the American Statistical Association 115(529), 393-402.
"""

from .cct import cct

__version__ = "0.1.0"
__all__ = ["cct"]
