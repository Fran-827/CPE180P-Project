# CPE180P Project: Fashion-MNIST ONNX Inference Service

A CNN trained on Fashion-MNIST, exported to ONNX, packaged in Docker, and served as a REST API on a simulated cloud VM (WSL2 Ubuntu). A Windows 11 PowerShell client operates it remotely over SSH.

## Architecture

    Windows 11 (client node)                     WSL2 Ubuntu (server node, "cloud VM")
    ------------------------                     -------------------------------------
    PowerShell + OpenSSH  --- SSH (key auth) --> sshd (password login disabled)
    client.py (HTTP POST) --- HTTP :8000 ------> Docker container (FastAPI + ONNX Runtime)
    deploy.ps1            --- SSH ------------>  systemd unit pulls image from Docker Hub

    GitHub push -> GitHub Actions (validate ONNX, pytest, build) -> Docker Hub -> VM pulls latest

## Results

| SLA | Target | Result |
|---|---|---|
| Model accuracy | >= 90% | see `docs/` (training output, about 92-94%) |
| Inference latency | <= 100 ms | see `/metrics` (sub-millisecond to a few ms) |

## Repository layout

| Path | Purpose |
|---|---|
| `train.py` | Train the CNN and export `model.onnx` |
| `check_onnx.py` | Validate the model with `onnx.checker` |
| `verify.py` | ONNX accuracy and latency check |
| `deploy/` | `app.py` (FastAPI), `Dockerfile`, `requirements.txt`, `model.onnx` |
| `client/client.py` | Windows client that sends JSON to `POST /v1/predict` |
| `tests/` | pytest suite for the API |
| `scripts/deploy.ps1` | Remote deployment script run from PowerShell |
| `scripts/demo-logs.sh` | tmux split screen: live logs and metrics |
| `.github/workflows/ci.yml` | CI/CD pipeline |

## API

| Route | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /v1/predict` | Body: `{"pixels": [784 numbers, 0-255]}`. Header `X-API-Key` required. Returns label, confidence, inference_ms, drift status |
| `GET /metrics` | Request count, p50/p95/max latency, drift status |

Input is validated: exactly 784 values, each between 0 and 255, otherwise HTTP 422.

**Drift alerting:** the service keeps a rolling window of the mean input pixel value. If it shifts more than 0.08 from the training mean (0.2860), it logs `DRIFT ALERT` and sets `drift.alert = true`.

## Run it locally

    docker run -d -p 8000:8000 -e API_KEY=choose-a-key --name fashion fran227/fashion-api:latest
    curl localhost:8000/health

## Remote operation from Windows

    # one-time: key-based SSH to the VM
    ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\sakura_key"
    # (public key is appended to ~/.ssh/authorized_keys on the VM)

    # run the client
    $env:API_KEY = "<key from /etc/fashion-api.env on the VM>"
    python client/client.py --url http://<VM_IP>:8000 --n 60
    python client/client.py --url http://<VM_IP>:8000 --n 40 --drift

    # redeploy the latest image
    .\scripts\deploy.ps1 <VM_IP>

## CI/CD

On every push to `main`, GitHub Actions validates the ONNX model, runs the tests, builds the image, and pushes `fran227/fashion-api:latest` to Docker Hub. `deploy.ps1` then restarts the systemd service on the VM, which pulls the new image.

## Security

- SSH public-key authentication only (`PasswordAuthentication no`, `PermitRootLogin no`)
- Host firewall (UFW) allowing only ports 22 and 8000
- Container runs as a non-root user
- Secrets (`API_KEY`, Docker Hub token) live in `.env`, `/etc/fashion-api.env` (mode 600), and GitHub Actions secrets. `.env` is git-ignored.
- Note: Docker-published ports bypass UFW, so UFW is a host-level control, not full protection for port 8000.
