import time, numpy as np, onnxruntime as ort
from torchvision import datasets

MEAN, STD = 0.2860, 0.3530
ds = datasets.FashionMNIST("data", train=False, download=True)
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

correct, times = 0, []
for i in range(1000):
    img, label = ds[i]
    x = ((np.array(img, dtype=np.float32) / 255 - MEAN) / STD).reshape(1, 1, 28, 28)
    t = time.perf_counter()
    out = sess.run(None, {"input": x})[0]
    times.append((time.perf_counter() - t) * 1000)
    correct += int(out.argmax() == label)

print(f"ONNX accuracy on 1000 test images: {correct/10:.1f}%")
print(f"Inference p50={np.percentile(times,50):.2f} ms, p95={np.percentile(times,95):.2f} ms")
