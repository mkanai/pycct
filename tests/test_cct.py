"""
Tests for the pycct package.

This module contains tests for the Cauchy Combination Test implementation.
"""

import pytest
import numpy as np
from pycct import cct

# Define a tolerance for floating-point comparisons
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-7  # Absolute tolerance


def test_basic_functionality():
    """Test basic functionality with the example from the R documentation."""
    pvalues = [0.02, 0.0004, 0.2, 0.1, 0.8]
    result = cct(pvalues)
    # Allow for some numerical differences
    expected = 0.001953404
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"


def test_nan_handling():
    """Test handling of NaN values."""
    # All values are NaN
    assert np.isnan(cct([np.nan, np.nan]))

    # Some values are NaN
    pvalues = [0.02, np.nan, 0.2, 0.1, 0.8]
    result = cct(pvalues)
    # Should ignore NaN and produce similar result to the original test
    # but with fewer p-values
    expected = 0.06614202
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"

    # Test with weights and NaN values
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    result = cct(pvalues, weights)
    # Weights should be adjusted automatically when NaNs are removed
    expected = 0.03075154
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"


def test_invalid_inputs():
    """Test behavior with invalid inputs."""
    # P-values outside [0, 1]
    with pytest.raises(ValueError, match="All p-values must be between 0 and 1!"):
        cct([-0.1, 0.5])

    with pytest.raises(ValueError, match="All p-values must be between 0 and 1!"):
        cct([0.1, 1.5])

    # Mismatched length of weights
    with pytest.raises(ValueError, match="The length of weights should be the same as that of the p-values!"):
        cct([0.1, 0.2, 0.3], [0.5, 0.5])

    # Negative weights
    with pytest.raises(ValueError, match="All the weights must be positive!"):
        cct([0.1, 0.2], [-0.5, 1.5])


def test_special_cases():
    """Test special cases mentioned in the R code."""
    # When p-value is 0
    assert cct([0.0, 0.5]) == 0.0

    # When p-value is 1
    result = cct([1.0, 0.5])
    expected = 0.5 * 2  # min(1, min(pvals) * len(pvals))
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"

    # When p-value is 1 but would exceed 1 after adjustment
    result = cct([1.0, 0.6, 0.7])
    expected = 1.0  # min(1, min(pvals) * len(pvals)) = min(1, 0.6 * 3) = 1
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"

    # Very small p-values (< 1e-16)
    small_pval = 1e-17
    result = cct([small_pval, 0.5])
    assert result < 1e-10, f"Expected very small value, got {result}"


def test_weights():
    """Test the effect of weights."""
    pvalues = [0.01, 0.5]

    # Equal weights (default)
    result_default = cct(pvalues)

    # Custom weights that heavily favor the first p-value
    result_weighted = cct(pvalues, [0.9, 0.1])

    # Result should be more significant (smaller) when we weight the more significant p-value higher
    assert result_weighted < result_default, f"Expected {result_weighted} < {result_default}"

    # Custom weights that heavily favor the second p-value
    result_weighted_opposite = cct(pvalues, [0.1, 0.9])

    # Result should be less significant (larger) when we weight the less significant p-value higher
    assert result_weighted_opposite > result_default, f"Expected {result_weighted_opposite} > {result_default}"


def test_input_types():
    """Test various input types."""
    # List input
    result_list = cct([0.1, 0.2])

    # Numpy array input
    result_numpy = cct(np.array([0.1, 0.2]))

    # Results should be the same
    assert np.isclose(result_list, result_numpy, rtol=RTOL, atol=ATOL), f"Expected {result_list}, got {result_numpy}"

    # Mixed input types
    result_mixed = cct([0.1, np.float64(0.2)])
    assert np.isclose(result_mixed, result_list, rtol=RTOL, atol=ATOL), f"Expected {result_list}, got {result_mixed}"


def test_extreme_values():
    """Test behavior with extreme values."""
    # Very large test statistic
    # This is a contrived example to trigger the special case in the code
    # where the test statistic is very large
    result = cct([1e-10], [1.0])
    assert result > 0, f"Expected positive value, got {result}"
    assert result < 1e-9, f"Expected very small value, got {result}"


def test_comparing_to_r_examples():
    """Test against some known examples from R."""
    # These values were calculated using the R implementation
    # Using the example in the R documentation
    pvalues = [0.02, 0.0004, 0.2, 0.1, 0.8]
    result = cct(pvalues)
    expected = 0.001953404
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"

    # With custom weights
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    result = cct(pvalues, weights)
    expected = 0.001901358
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"


def test_edge_case_arrays():
    """Test edge cases for arrays."""
    # Empty array
    assert np.isnan(cct([]))

    # Single value
    result = cct([0.5])
    expected = 0.5
    assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Expected {expected}, got {result}"

    # Very large arrays
    np.random.seed(42)  # For reproducibility
    large_pvals = np.random.uniform(0, 1, 1000)
    result = cct(large_pvals)
    assert 0 <= result <= 1, f"Expected value between 0 and 1, got {result}"
