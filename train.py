import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN, STD = 0.2860, 0.3530

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 3 * 3, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=512)

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def evaluate():
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(test_ds)

for epoch in range(12):
    model.train()
    for x, y in train_dl:
        opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    print(f"epoch {epoch+1}: test acc = {evaluate():.4f}")

model.eval()
torch.onnx.export(
    model, torch.randn(1, 1, 28, 28), "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)
print("Exported model.onnx")
