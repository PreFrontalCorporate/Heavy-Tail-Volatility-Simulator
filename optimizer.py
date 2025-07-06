import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def run_portfolio_optimization(Sigma_list, mu_list, rho, lambda_sparse):
    Sigma = np.array(Sigma_list)
    mu = np.array(mu_list)

    n = len(mu)
    w = cp.Variable(n)

    # Robustified objective
    risk = cp.quad_form(w, Sigma)
    robust_penalty = rho * cp.norm(w, 2)
    sparsity_penalty = lambda_sparse * cp.norm1(w)
    objective = cp.Minimize(risk + robust_penalty - mu @ w + sparsity_penalty)

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    weights = w.value.tolist()
    num_active = sum(np.array(weights) > 1e-4)

    # Plotting
    fig, ax = plt.subplots()
    ax.stem(range(n), weights, markerfmt='bo')
    ax.set_title("Optimized Portfolio Weights")
    ax.set_xlabel("Asset Index")
    ax.set_ylabel("Weight")

    result_summary = {
        "objective_value": prob.value,
        "num_active_weights": num_active,
        "weights": weights
    }

    return result_summary, fig
