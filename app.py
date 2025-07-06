# heavy_tail_app/app.py

from flask import Blueprint, request, jsonify, send_file
import numpy as np
import cvxpy as cp

heavy_tail_bp = Blueprint("heavy_tail", __name__)

def perform_optimization(mu_list, Sigma_list, l1_lambda):
    mu = np.array(mu_list)
    Sigma = np.array(Sigma_list)
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.quad_form(w, Sigma) - mu @ w + l1_lambda * cp.norm1(w)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    weights = w.value.tolist()
    num_active_weights = sum(abs(wi) > 1e-4 for wi in weights)
    return {"weights": weights, "num_active_weights": num_active_weights, "objective_value": float(prob.value)}

@heavy_tail_bp.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    mu = data["mu"]
    Sigma = data["Sigma"]
    l1_lambda = data["l1_lambda"]
    result = perform_optimization(mu, Sigma, l1_lambda)
    return jsonify(result)

@heavy_tail_bp.route("/download_plot", methods=["GET"])
def download_plot():
    return send_file("static/example_plot.png", mimetype="image/png")
