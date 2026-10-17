import logging, os, time
from collections import deque
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator

MEAN, STD = 0.2860, 0.3530          # training-set pixel statistics
DRIFT_THRESHOLD = 0.08               # alert if rolling mean shifts more than this
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
API_KEY = os.getenv("API_KEY")       # secret comes from the environment, never from code

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Fashion-MNIST Classifier", version="1.0")
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
sess.run(None, {"input": np.zeros((1, 1, 28, 28), dtype=np.float32)})  # warm-up
latencies = deque(maxlen=1000)
input_means = deque(maxlen=50)

class Payload(BaseModel):
    pixels: list[float]

    @field_validator("pixels")
    @classmethod
    def check(cls, v):
        if len(v) != 784:
            raise ValueError("pixels must contain exactly 784 values (28x28)")
        if any(not (0 <= p <= 255) for p in v):   # also rejects NaN
            raise ValueError("pixel values must be numbers between 0 and 255")
        return v

def drift_status():
    if len(input_means) < 20:
        return {"alert": False, "window": len(input_means), "mean_shift": None}
    shift = abs(float(np.mean(input_means)) - MEAN)
    return {"alert": shift > DRIFT_THRESHOLD, "window": len(input_means),
            "mean_shift": round(shift, 4)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/predict")
def predict(p: Payload, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        log.warning("rejected request: bad API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    raw = np.array(p.pixels, dtype=np.float32) / 255
    x = ((raw - MEAN) / STD).reshape(1, 1, 28, 28)
    t = time.perf_counter()
    logits = sess.run(None, {"input": x})[0][0]
    ms = (time.perf_counter() - t) * 1000
    latencies.append(ms)
    input_means.append(float(raw.mean()))
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    idx = int(probs.argmax())
    drift = drift_status()
    log.info("predict label=%s conf=%.3f inference_ms=%.2f", CLASSES[idx], probs[idx], ms)
    if drift["alert"]:
        log.warning("DRIFT ALERT: input mean shifted by %.3f (threshold %.2f)",
                    drift["mean_shift"], DRIFT_THRESHOLD)
    return {"label": CLASSES[idx], "confidence": round(float(probs[idx]), 4),
            "inference_ms": round(ms, 3), "drift": drift}

@app.get("/metrics")
def metrics():
    if not latencies:
        return {"requests": 0, "drift": drift_status()}
    a = np.array(latencies)
    return {"requests": len(a), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "max_ms": round(float(a.max()), 3), "drift": drift_status()}
