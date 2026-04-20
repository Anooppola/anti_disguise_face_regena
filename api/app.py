"""
api/app.py - FastAPI backend for Anti-Disguise GAN inference
Endpoints:
  GET  /           → health check
  GET  /info       → model info
  POST /predict    → masked image → reconstructed face
"""

import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from project root .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Anti-Disguise Face Reconstruction API",
    description=(
        "Upload a masked/occluded face image and receive "
        "a GAN-reconstructed unmasked face."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Streamlit and browser access
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ──────────────────────────────────────────────────────────────

engine = None   # InferenceEngine (loaded at startup)


@app.on_event("startup")
async def startup_event():
    """Load the GAN model once when the server starts."""
    global engine
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.inference import InferenceEngine

    model_path = os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[1] / "saved_models" / "generator_best.pth")
    )

    if not Path(model_path).exists():
        logger.warning("Model file not found at %s. /predict will return 503.", model_path)
        return

    try:
        engine = InferenceEngine(model_path=model_path)
        logger.info("✅ Model loaded from: %s", model_path)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "message": "Anti-Disguise Face Reconstruction API is running.",
    }


@app.get("/info", tags=["info"])
def model_info():
    return {
        "architecture": "Pix2Pix GAN (U-Net Generator + PatchGAN Discriminator)",
        "input":        "256×256 RGB masked-face image",
        "output":       "256×256 RGB reconstructed unmasked face",
        "losses":       ["Adversarial (LSGAN)", "L1", "Perceptual (VGG19)"],
    }


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a masked face image → returns a PNG of the reconstructed face.
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs and MODEL_PATH environment variable.",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received content_type: {content_type}",
        )

    try:
        contents = await file.read()
        logger.info("Received '%s' (%d bytes)", file.filename, len(contents))

        result_bytes = engine.predict_bytes(contents, fmt="PNG")

        logger.info("✅ Inference complete for '%s'", file.filename)
        return Response(content=result_bytes, media_type="image/png")

    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


# ─── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
