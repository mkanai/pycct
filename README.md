# PyCCT: Cauchy Combination Test for Python

PyCCT is a Python implementation of the Cauchy Combination Test (CCT), a powerful p-value aggregation method using the Cauchy distribution.

## Installation

```bash
# TODO: pip install pycct
pip install git+https://github.com/mkanai/pycct
```

## Usage

```python
import numpy as np
from pycct import cct

# Example with equal weights
pvalues = [0.02, 0.0004, 0.2, 0.1, 0.8]
result = cct(pvalues)
print(f"Combined p-value: {result}")

# Example with custom weights
weights = [0.5, 0.2, 0.1, 0.1, 0.1]
result = cct(pvalues, weights)
print(f"Combined p-value with weights: {result}")

# Example with missing values
pvalues_with_na = [0.02, np.nan, 0.2, 0.1, 0.8]
result = cct(pvalues_with_na)
print(f"Combined p-value ignoring NaN: {result}")
```

## Features

- Combines p-values using the Cauchy method
- Handles NaN values in the input
- Supports custom weights for p-values
- Special handling for extreme cases (p-values of 0 or 1)

## Reference

Liu, Y., & Xie, J. (2020). Cauchy combination test: a powerful test with analytic p-value calculation under arbitrary dependency structures. _Journal of the American Statistical Association 115_(529), 393-402.
