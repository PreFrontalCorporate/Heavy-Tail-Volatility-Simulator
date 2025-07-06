import json
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_optimize_endpoint(client):
    data = {
        "mu": [0.01, 0.02, 0.015],
        "Sigma": [
            [0.005, -0.010, 0.004],
            [-0.010, 0.040, -0.002],
            [0.004, -0.002, 0.023]
        ],
        "l1_lambda": 0.01
    }
    response = client.post("/optimize", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200

    result = response.get_json()
    assert "weights" in result
    assert "num_active_weights" in result
    assert "objective_value" in result
    assert abs(sum(result["weights"]) - 1.0) < 1e-3

def test_download_plot(client):
    response = client.get("/download_plot")
    assert response.status_code == 200
    assert response.content_type == "image/png"
