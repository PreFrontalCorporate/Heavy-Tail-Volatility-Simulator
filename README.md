# Heavy-Tail Volatility Simulator API

**CBB/Seas Product Line — Robust Sparse Portfolio API**

## Overview

This Flask API enables users to run heavy-tail robust portfolio optimization with optional sparsity constraints, inspired by advanced dual conic formulations. It is designed for research, advanced investment analysis, and integration into CBB/Seas risk products.

## Endpoints

### `/`

Basic health check endpoint to verify API is live.

### `/optimize` (POST)

Runs portfolio optimization using robust and sparse formulations.

#### JSON Body

```
{
    "Sigma": [[...], [...]],
    "mu": [...],
    "rho": 0.1,
    "lambda_sparse": 0.01
}
```

* `Sigma`: Covariance matrix of assets (2D list)
* `mu`: Expected return vector (list)
* `rho`: Robustness penalty parameter (default: 0.1)
* `lambda_sparse`: Sparsity penalty parameter (default: 0.01)

#### Response

```
{
    "objective_value": float,
    "num_active_weights": int,
    "weights": [...]
}
```

### `/download_plot`

Returns a visualization (PNG) of the optimized weights.

## Setup & Usage

```
pip install -r requirements.txt
python app.py
```

Send POST requests to `/optimize` with JSON data to get portfolio allocations and robust sparse analysis.

## Future Directions

* Integrate exact cardinality constraints using mixed-integer programming (MIP)
* Extend to heavy-tail risk metrics (CVaR, EVT-based measures)
* Add user authentication and session tracking for production deployments
* Provide automatic PDF report generation

## Academic & Mathematical Background

The core model is based on minimizing a robustified quadratic risk term with L1 sparsity promotion:

```
min_w   wᵀ Σ w + ρ ||w||₂ - μᵀw + λ_sparse ||w||₁
```

Subject to:

```
sum(w) = 1
w ≥ 0
```

This formulation combines:

* Classical Markowitz-type quadratic risk
* Robust perturbations (via 2-norm penalty)
* L1 sparsity for interpretable, concentrated portfolios

It is solved via conic convex optimization using CVXPY (with fallback to SCS solver), enabling practical approximate solutions even when high-precision solvers are unavailable.

## License & Attribution

CBB/Seas (c) 2025. All rights reserved.

---

For advanced theoretical notes and proofs, refer to `docs/theory_notes.pdf` (to be included).
