import onnx
m = onnx.load("deploy/model.onnx")
onnx.checker.check_model(m)
print("onnx.checker: model.onnx is valid")
