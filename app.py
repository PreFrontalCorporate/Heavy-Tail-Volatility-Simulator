from flask import Flask, request, jsonify, send_file
import numpy as np
import cvxpy as cp

app = Flask(__name__)

def perform_optimization(mu_list, Sigma_list, l1_lambda):
    # Convert lists to NumPy arrays
    mu = np.array(mu_list)
    Sigma = np.array(Sigma_list)

    n = len(mu)
    w = cp.Variable(n)

    # Objective: minimize variance minus return plus L1 sparsity
    obj = cp.quad_form(w, Sigma) - mu @ w + l1_lambda * cp.norm1(w)

    # Constraints: sum to 1, no short selling
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)

    # Convert to native Python types
    weights = w.value.tolist()
    weights = [float(x) for x in weights]
    num_active_weights = sum([abs(wi) > 1e-4 for wi in weights])

    return {
        "weights": weights,
        "num_active_weights": num_active_weights,
        "objective_value": float(prob.value)
    }

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    mu = data["mu"]
    Sigma = data["Sigma"]
    l1_lambda = data["l1_lambda"]

    result = perform_optimization(mu, Sigma, l1_lambda)

    return jsonify(result)

@app.route("/download_plot", methods=["GET"])
def download_plot():
    return send_file("static/example_plot.png", mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
