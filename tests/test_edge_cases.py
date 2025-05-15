"""
Additional tests focusing on edge cases and validation against the R implementation.
"""

import pytest
import numpy as np
from pycct import cct

# Define a tolerance for floating-point comparisons
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-7  # Absolute tolerance


def test_empty_array_behavior():
    """Test behavior with empty arrays."""
    # Empty arrays should return NaN since all elements are considered NaN
    assert np.isnan(cct([]))
    assert np.isnan(cct(np.array([])))


def test_single_pvalue():
    """Test behavior with a single p-value."""
    # Single p-value should return itself
    test_cases = [0.5, 0.1, 0.0, 1.0]
    for p in test_cases:
        result = cct([p])
        assert np.isclose(result, p, rtol=RTOL, atol=ATOL), f"Expected {p}, got {result}"


def test_all_identical_pvalues():
    """Test behavior when all p-values are identical."""
    # All identical values
    test_cases = [0.5, 0.1, 0.0, 1.0]
    for p in test_cases:
        result = cct([p, p, p])
        assert np.isclose(result, p, rtol=RTOL, atol=ATOL), f"Expected {p}, got {result}"


def test_mixed_extreme_values():
    """Test behavior with mixed extreme values."""
    # Mix of 0 and other values
    result = cct([0.0, 0.5, 0.8])
    expected = 0.0
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"

    # Mix of 1 and other values
    result = cct([1.0, 0.1, 0.2])
    expected = 0.1 * 3  # min(1, min(pvals) * len(pvals))
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"


def test_weight_normalization():
    """Test that weights are properly normalized."""
    pvals = [0.2, 0.3]

    # These should give identical results as weights are normalized
    result1 = cct(pvals, [1, 1])
    result2 = cct(pvals, [0.5, 0.5])
    result3 = cct(pvals, [100, 100])

    assert np.isclose(result1, result2, rtol=RTOL, atol=ATOL), f"Expected {result1}, got {result2}"
    assert np.isclose(result1, result3, rtol=RTOL, atol=ATOL), f"Expected {result1}, got {result3}"


def test_weight_extreme_cases():
    """Test behavior with extreme weights."""
    pvals = [0.1, 0.9]

    # All weight on first p-value
    result1 = cct(pvals, [1, 0])
    expected1 = 0.1  # Should be equal to first p-value
    assert np.isclose(result1, expected1, rtol=RTOL, atol=ATOL), f"Expected {expected1}, got {result1}"

    # All weight on second p-value
    result2 = cct(pvals, [0, 1])
    expected2 = 0.9  # Should be equal to second p-value
    assert np.isclose(result2, expected2, rtol=RTOL, atol=ATOL), f"Expected {expected2}, got {result2}"

    # Extremely unbalanced weights
    result3 = cct(pvals, [1e10, 1])
    # Should be very close to first p-value but not exactly
    assert np.isclose(result3, 0.1, rtol=RTOL, atol=ATOL), f"Expected close to 0.1, got {result3}"


def test_known_r_values():
    """Test against known values from R for validation."""
    # Values calculated using the R implementation
    test_cases = [
        {"pvals": [0.5, 0.5], "weights": None, "expected": 0.5},
        {"pvals": [0.01, 0.02, 0.03], "weights": None, "expected": 0.01636684},
        {"pvals": [0.01, 0.02, 0.03], "weights": [0.6, 0.3, 0.1], "expected": 0.01276716},
        {"pvals": [0.001, 0.005, 0.01, 0.05, 0.1], "weights": None, "expected": 0.003760775},
        {"pvals": [0.001, 0.005, 0.01, 0.05, 0.1], "weights": [0.5, 0.2, 0.1, 0.1, 0.1], "expected": 0.001808488},
    ]

    for case in test_cases:
        result = cct(case["pvals"], case["weights"])
        expected = case["expected"]
        assert np.isclose(
            result, expected, rtol=RTOL, atol=ATOL
        ), f"Test case {case['pvals']}: Expected {expected}, got {result}"


def test_very_small_pvalues():
    """Test behavior with very small p-values that aren't quite zero."""
    small_pvals = [1e-100, 1e-200]

    # Should handle very small p-values correctly
    result = cct(small_pvals)
    expected = 2e-200  # Expected value based on CCT calculation
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"
