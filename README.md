# Zero-Shot Anti-Disguise Facial Reconstruction using GANs (Production Edition)

## Project Goal
This deep learning project builds a system that reconstructs hidden facial regions when a face is partially occluded. As part of a final academic evaluation, this project integrates best practices in **DevOps and MLOps**. The core trained Generative Adversarial Network (GAN) has been wrapped in a production-ready FastAPI architecture, fully containerized, tracked with MLflow, and monitored with CI/CD.

## 🏗️ System Architecture

```text
       [USER] (Web / Mobile / cURL)
          │
          ▼
   [FastAPI Application] ──(Loads .env)──> [PyTorch Model weights]
          │
      --------- (Docker Container) ---------
          │
          ▼
   [Inference Engine] -----> Returns Reconstructed Face Image
          ▲
          │
  (GitHub Actions CI/CD) -> Automatically tests API & Builds Docker image on push
          │
          ▼
    [MLflow Tracking] -----> Tracks Experiments, Logs Losses & PSNR/SSIM metrics during Training
```

## 🚀 Features (DevOps & MLOps Highlights)
- **FastAPI**: Synchronous & async serving for high performance.
- **Dockerized**: Containerized environment for guaranteed reproducibility.
- **Continuous Integration**: GitHub actions validate code with PyTest before deploying.
- **Environment Management**: Secrets injected via `.env`.
- **MLflow Tracking**: Training hyper-parameters, loss graphs, and models are tracked implicitly.
- **Structured Logging**: Standard Python logging logs inference latencies and model loading exceptions.

---

## 🛠️ Step-by-step Local Run Instructions

### 1. Setup Environment
Ensure Python 3.10 is installed.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure `.env`
Create a `.env` file in the root if not present:
```env
MODEL_PATH=saved_models/generator_best.pth
PORT=8000
HOST=0.0.0.0
```

### 3. Run the FastAPI Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
You can view the interactive Swagger API docs at: `http://localhost:8000/docs`

---

## 🐳 Running with Docker

Instead of installing Python dependencies manually, you can run the entire solution via Docker!

### Method A: Docker Compose (Recommended)
```bash
docker-compose up --build -d
```
The API is instantly available at `http://localhost:8000`. Stop it using `docker-compose down`.

### Method B: Native Docker
```bash
docker build -t antidisguise_api:latest .
docker run -p 8000:8000 --env-file .env antidisguise_api:latest
```

---

## 🔗 API Testing & Usage (Viva Demonstration)

The API exposes a POST `/predict` endpoint that accepts multipart form data.

### Sample Input JSON
*(Note: Because the payload accepts a file, JSON is not used directly, but equivalent metadata representation looks like:)*
```json
{
  "file": "<Binary Image Data - test.jpg>"
}
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@input.png" --output "reconstructed.png"
```

### Postman Example
1. Open Postman, select **POST** and enter `http://localhost:8000/predict`
2. Go to the **Body** tab, select **form-data**.
3. Create a Key named `file`, change the Type from *Text* to *File*.
4. Upload your test image into the Value field and click Send. The response will be the unmasked image!

---

## ☁️ Cloud Deployment Guide (EC2 / Render)

### Option 1: Render / Railway (Easiest)
1. Fork this repository to your GitHub account.
2. Go to **Render.com** (or Railway.app), click **New Web Service**.
3. Point to your repository branch. Ensure the environment is set to `Docker`.
4. Add Environment Variables: `MODEL_PATH=saved_models/generator_best.pth`.
5. Click **Deploy**. The service will build and provide a public HTTPS URL seamlessly mapping Port 8000.

### Option 2: AWS EC2 (Manual)
1. Launch an Ubuntu 22.04 t2.medium instance with public HTTP access on port 8000.
2. SSH into your instance.
3. Install Docker and clone the repository.
4. Run `sudo docker-compose up --build -d`.
5. Access your API via `http://<EC2-PUBLIC-IP>:8000`.

---

## 🧪 MLOps: Model Tracking (MLflow)

To view training performance metrics:
1. Open a terminal and type: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
2. Visit `http://127.0.0.1:5000`
3. The UI will show standard metrics:
   - **Hyperparameters:** `learning_rate`, `batch_size`, `lambda_perceptual`.
   - **Metrics:** `G_loss` (Generator Error), `D_loss` (Detector Error) evaluated sequentially per epoch.
   - **Artifacts:** Checkpoints logged as generic PyTorch traces.

### Unit Tests
Execute the testing pipeline independently using PyTest:
```bash
pytest tests/
```

---
## Ethical Note
**Important:** This system is designed for **research purposes and security enhancement only**. All users must strictly adhere to privacy regulations and laws regarding biometric data and facial generation. Misuse for identity falsification or unauthorized surveillance is strictly condemned.
