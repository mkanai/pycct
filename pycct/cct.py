"""
Cauchy Combination Test implementation in Python.

This implementation is based on the original R code from the STAR package:
https://github.com/xihaoli/STAAR/blob/dc4f7e509f4fa2fb8594de48662bbd06a163108c/R/CCT.R
and the modifided version from the SAIGEQTL package:
https://github.com/weizhou0/qtl/blob/1f100118df51a808fb36780325dab54298e78a3a/R/CCT_modified.R
"""

import numpy as np
from typing import Optional, Union, List, Tuple


def cct(pvalues: Union[List[float], np.ndarray], weights: Optional[Union[List[float], np.ndarray]] = None) -> float:
    """
    Cauchy Combination Test for p-value aggregation.

    This function takes a numeric array of p-values and a numeric array of non-negative weights,
    and returns the aggregated p-value using the Cauchy method.

    Parameters
    ----------
    pvalues : array-like
        A numeric array of p-values, where each element is between 0 and 1.
    weights : array-like, optional
        A numeric array of non-negative weights. If None, equal weights are assumed.

    Returns
    -------
    float
        The aggregated p-value combining p-values from the input array.

    Examples
    --------
    >>> cct([0.02, 0.0004, 0.2, 0.1, 0.8])
    0.004536...

    References
    ----------
    Liu, Y., & Xie, J. (2020). Cauchy combination test: a powerful test
    with analytic p-value calculation under arbitrary dependency structures.
    Journal of the American Statistical Association 115(529), 393-402.
    """
    # Convert inputs to numpy arrays for efficient operations
    pvals = np.asarray(pvalues, dtype=float)

    # Handle NaN values (equivalent to R's is.na)
    notna_mask = ~np.isnan(pvals)

    # Return NA (np.nan) if all values are NA
    if not np.any(notna_mask):
        return np.nan

    # Filter out NA values
    pvals = pvals[notna_mask]

    # Always convert weights to numpy array if provided
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        # Adjust weights if we filtered some p-values
        if np.sum(~notna_mask) > 0:
            weights = weights[notna_mask]

    # Check if there are values outside [0, 1]
    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be between 0 and 1!")

    # Check if there are p-values that are exactly 0 or 1
    is_zero = np.any(pvals == 0)
    is_one = np.any(pvals == 1)

    if is_zero:
        return 0.0

    if is_one:
        # When individual p-value = 1, use minimum p-value
        # (modified from the original R code as per the comment)
        return min(1.0, np.min(pvals) * len(pvals))

    # Handle weights
    if weights is None:
        # Equal weights
        weights = np.ones_like(pvals) / len(pvals)
    else:
        # Ensure weights is a numpy array
        weights = np.asarray(weights, dtype=float)

        # Check weights length
        if len(weights) != len(pvals):
            raise ValueError("The length of weights should be the same as that of the p-values!")

        # Check if all weights are non-negative
        if np.any(weights < 0):
            raise ValueError("All the weights must be positive!")

        # Standardize weights
        weights = weights / np.sum(weights)

    # Check if there are very small non-zero p-values
    is_small = pvals < 1e-16

    if not np.any(is_small):
        cct_stat = np.sum(weights * np.tan((0.5 - pvals) * np.pi))
    else:
        cct_stat = np.sum((weights[is_small] / pvals[is_small]) / np.pi)
        cct_stat += np.sum(weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * np.pi))

    # Check if the test statistic is very large
    if cct_stat > 1e15:
        pval = (1 / cct_stat) / np.pi
    else:
        # Use the survival function (1 - CDF) of the Cauchy distribution
        pval = 1 - _pcauchy(cct_stat)

    return pval


def _pcauchy(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Distribution function of the standard Cauchy distribution.

    This is equivalent to R's pcauchy function.

    Parameters
    ----------
    x : float or array-like
        Quantiles.

    Returns
    -------
    float or array-like
        Probabilities.
    """
    return 0.5 + np.arctan(x) / np.pi
