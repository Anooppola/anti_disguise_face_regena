# 🎭 Anti-Disguise Face Reconstruction — DevOps + MLOps Pipeline

> **Pix2Pix GAN** system that reconstructs unmasked faces from masked/occluded inputs — with a complete production-grade DevOps and MLOps stack.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)  
2. [Architecture](#-architecture)  
3. [Project Structure](#-project-structure)  
4. [Quick Start](#-quick-start)  
5. [Docker Usage](#-docker-usage)  
6. [MLflow Tracking](#-mlflow-tracking)  
7. [API Usage](#-api-usage)  
8. [Frontend Usage](#-frontend-usage)  
9. [CI/CD Pipeline](#-cicd-pipeline)  
10. [Training](#-training)  
11. [Evaluation Metrics](#-evaluation-metrics)  

---

## 🔍 Project Overview

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch — Pix2Pix GAN |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Metrics | PSNR, SSIM, Precision, Recall, F1 |

---

## 🏗 Architecture

```
Masked Face (256×256)
        │
        ▼
 ┌──────────────┐     adversarial    ┌─────────────────────┐
 │  U-Net       │ ◄───────────────── │  PatchGAN           │
 │  Generator   │                    │  Discriminator 70×70│
 └──────┬───────┘                    └─────────────────────┘
        │ L1 + Perceptual (VGG19) + Adversarial
        ▼
Reconstructed Face (256×256)
```

**Loss functions:**
- `L_G = L_adv + λ_L1 × L_L1 + λ_percep × L_percep`
- `L_adv` — LSGAN (MSE-based adversarial)
- `L_L1` — Pixel-wise MAE (λ=100)
- `L_percep` — VGG19 feature MAE (λ=10)

---

## 📁 Project Structure

```
anti-disguise-mlops/
├── data/
│   ├── masked/          ← M0001.png, M0002.png …
│   └── unmasked/        ← UM0001.png, UM0002.png …
├── src/
│   ├── model.py         ← Generator + Discriminator + VGGPerceptualLoss
│   ├── data_loader.py   ← Paired dataset + DataLoader factory
│   ├── preprocessing.py ← Image pre/post-processing utilities
│   ├── train.py         ← Full training loop with MLflow
│   ├── evaluate.py      ← PSNR, SSIM, F1 metrics
│   └── inference.py     ← InferenceEngine class
├── mlflow_utils/
│   ├── mlflow_utils.py  ← setup/log helpers
│   ├── track_experiments.py ← test-run script
│   └── run_mlflow_ui.py ← launch MLflow server
├── api/
│   └── app.py           ← FastAPI backend
├── frontend/
│   └── streamlit_app.py ← Streamlit UI
├── docker/
│   ├── Dockerfile       ← Multi-stage (api + frontend)
│   └── docker-compose.yml
├── .github/workflows/
│   └── ci-cd.yml        ← lint → test → docker build/push
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_evaluate.py
├── saved_models/        ← place generator_best.pth here
├── requirements.txt
├── .env
├── .gitignore
└── main.py              ← CLI entry point
```

---

## ⚡ Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/<you>/anti-disguise-mlops.git
cd anti-disguise-mlops
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Add your dataset

```
data/masked/    → M0001.png, M0002.png …
data/unmasked/  → UM0001.png, UM0002.png …
```

Files are paired by the numeric ID in the filename.

### 3. Train

```bash
python main.py train --epochs 50 --batch_size 4
```

### 4. Log a test MLflow experiment (no GPU needed)

```bash
python main.py test-exp
```

### 5. Open MLflow UI

```bash
python main.py mlflow
# → http://localhost:5000
```

### 6. Start the API

```bash
python main.py serve
# → http://localhost:8000/docs
```

### 7. Start the frontend

```bash
python main.py frontend
# → http://localhost:8501
```

---

## 🐳 Docker Usage

### Start all services with 1 command

```bash
# From project root
cd docker
docker-compose up --build -d
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| MLflow  | http://localhost:5000 |
| Streamlit | http://localhost:8501 |

### Stop all services

```bash
docker-compose down
```

### View logs

```bash
docker-compose logs -f api
docker-compose logs -f mlflow
docker-compose logs -f frontend
```

> **Note:** Place your trained `generator_best.pth` in `saved_models/` before starting.  
> It is volume-mounted into the API container automatically.

---

## 📊 MLflow Tracking

| Logged parameter | Key |
|-----------------|-----|
| Learning rate | `learning_rate` |
| Batch size | `batch_size` |
| Epochs | `epochs` |
| L1 weight | `lambda_l1` |
| Perceptual weight | `lambda_percep` |

| Logged metric | Description |
|--------------|-------------|
| `g_loss` | Generator total loss |
| `d_loss` | Discriminator loss |
| `psnr` | Peak Signal-to-Noise Ratio (dB) |
| `ssim` | Structural Similarity Index |
| `f1_score` | Discriminator F1 |
| `best_psnr` | Best validation PSNR |

---

## 🔌 API Usage

### Health check

```bash
curl http://localhost:8000/
```

### Model info

```bash
curl http://localhost:8000/info
```

### Predict (reconstruct face)

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@masked_face.png" \
     --output reconstructed.png
```

### Python example

```python
import requests

with open("masked_face.png", "rb") as f:
    r = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("input.png", f, "image/png")},
    )

with open("reconstructed.png", "wb") as out:
    out.write(r.content)
```

---

## 🖥 Frontend Usage

1. Open **http://localhost:8501**
2. Upload a masked face image (PNG/JPG)
3. Click **🚀 Reconstruct Face**
4. View the **before/after comparison**
5. Download the result with **⬇️ Download Result**

---

## 🔄 CI/CD Pipeline

On every push to `main` or PR:

```
Push / PR
   │
   ▼
┌──────┐    ┌──────┐    ┌────────────────────┐
│ Lint │ →  │ Test │ →  │ Docker Build & Push │
│flake8│    │pytest│    │ (main branch only)  │
│black │    │      │    │ → ghcr.io registry  │
└──────┘    └──────┘    └────────────────────┘
```

---

## 🏋 Training

```bash
python main.py train \
  --masked_dir   data/masked \
  --unmasked_dir data/unmasked \
  --epochs       100 \
  --batch_size   4 \
  --lr           0.0002 \
  --lambda_l1    100 \
  --lambda_percep 10 \
  --save_every   10
```

Model checkpoints are saved to `saved_models/` every 10 epochs.  
The best model by PSNR is saved as `generator_best.pth`.

---

## 📐 Evaluation Metrics

```bash
python main.py evaluate \
  --model        saved_models/generator_best.pth \
  --masked_dir   data/masked \
  --unmasked_dir data/unmasked
```

| Metric | Description | Target |
|--------|------------|--------|
| PSNR | Peak Signal-to-Noise Ratio | > 25 dB |
| SSIM | Structural Similarity | > 0.75 |
| Precision | Discriminator precision | → 0.5 (balanced) |
| Recall | Discriminator recall | → 0.5 (balanced) |
| F1 | Harmonic mean | → 0.5 |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
