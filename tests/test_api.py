import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), "..", "deploy"))
sys.path.insert(0, os.getcwd())
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json() == {"status": "ok"}

def test_predict_valid():
    r = client.post("/v1/predict", json={"pixels": [0.0] * 784})
    assert r.status_code == 200
    assert r.json()["label"] in ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def test_predict_wrong_length():
    assert client.post("/v1/predict", json={"pixels": [0.0] * 10}).status_code == 422

def test_predict_out_of_range():
    assert client.post("/v1/predict", json={"pixels": [999.0] * 784}).status_code == 422

def test_metrics():
    assert "drift" in client.get("/metrics").json()
