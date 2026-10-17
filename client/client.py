import argparse, glob, os, time
import numpy as np, requests
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--drift", action="store_true", help="send bright noise to trigger the drift alert")
a = ap.parse_args()

headers = {"X-API-Key": os.getenv("API_KEY", "")}
files = sorted(glob.glob("samples/*.png"))
times = []
for i in range(a.n):
    if a.drift:
        pixels = np.random.randint(100, 256, 784).tolist()
    else:
        img = Image.open(files[i % len(files)]).convert("L").resize((28, 28))
        pixels = np.asarray(img).flatten().tolist()
    t = time.perf_counter()
    r = requests.post(f"{a.url}/v1/predict", json={"pixels": pixels}, headers=headers, timeout=5)
    ms = (time.perf_counter() - t) * 1000
    times.append(ms)
    d = r.json()
    print(f"{i+1:3d} HTTP {r.status_code} {d.get('label')} conf={d.get('confidence')} "
          f"rtt={ms:.1f}ms drift_alert={d.get('drift', {}).get('alert')}")
print(f"\nRound trip p50={np.percentile(times,50):.1f} ms, p95={np.percentile(times,95):.1f} ms")
