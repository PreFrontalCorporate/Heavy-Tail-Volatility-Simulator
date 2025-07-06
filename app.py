from flask import Flask, request, jsonify, send_file
from optimizer import run_portfolio_optimization
import io

app = Flask(__name__)

@app.route("/")
def index():
    return "Heavy-Tail Volatility Simulator API (CBB/Seas Product Line)"

@app.route("/optimize", methods=["POST"])
def optimize():
    """
    Expects JSON with:
    {
        "Sigma": [[...], [...], ...],
        "mu": [...],
        "rho": float,
        "lambda_sparse": float
    }
    """
    data = request.get_json()

    Sigma = data.get("Sigma")
    mu = data.get("mu")
    rho = data.get("rho", 0.1)
    lambda_sparse = data.get("lambda_sparse", 0.01)

    result, plot_img = run_portfolio_optimization(Sigma, mu, rho, lambda_sparse)

    # Convert plot to bytes
    buf = io.BytesIO()
    plot_img.savefig(buf, format="png")
    buf.seek(0)

    response = {
        "objective_value": float(result["objective_value"]),
        "num_active_weights": int(result["num_active_weights"]),
        "weights": [float(w) for w in result["weights"]]
    }

    return jsonify(response)

@app.route("/download_plot", methods=["GET"])
def download_plot():
    return send_file("static/example_plot.png", mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

