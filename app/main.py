import os
import sys
import logging
from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv
import torch
from torchvision import transforms
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path to resolve 'models' correctly
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from models.generator import Generator
except ImportError as e:
    logger.error(f"Failed to import Generator. Ensure your paths are correct: {e}")
    raise e

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

app = FastAPI(
    title="Anti-Disguise Face Reconstruction API",
    description="API for removing face masks using a Pix2Pix GAN model.",
    version="1.0.0"
)

# Global variables
model = None
device = None
transform = None

@app.on_event("startup")
async def startup_event():
    global model, device, transform
    logger.info("Initializing application and loading model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model_path = os.getenv("MODEL_PATH", "saved_models/generator_best.pth")
    full_model_path = str(BASE_DIR / model_path)
    
    if not os.path.exists(full_model_path):
        logger.error(f"Model file not found at {full_model_path}")
    else:
        try:
            model = Generator(in_channels=3, out_channels=3).to(device)
            model.load_state_dict(torch.load(full_model_path, map_location=device, weights_only=True))
            model.eval()
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model from {full_model_path}: {e}")
            raise e
            
    # Define image transformations (must match test.py and training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
    ])

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Anti-Disguise Face Reconstruction API is running."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (masked face), passes it through the GAN,
    and returns the unmasked image as a standard image response.
    """
    if model is None:
        logger.error("Prediction attempted but model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
        
    try:
        logger.info(f"Received file for prediction: {file.filename} (content_type: {file.content_type})")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        logger.info("Running inference...")
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Postprocess: convert tensor [-1, 1] back to Image [0, 255]
        output_tensor = (output_tensor.squeeze(0).cpu() + 1) / 2.0  # normalize to [0,1]
        output_tensor = output_tensor.clamp(0, 1)
        
        # Convert to PIL Image
        output_image = transforms.ToPILImage()(output_tensor)
        
        # Save to buffer
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        
        logger.info("✅ Inference successful. Returning unmasked image.")
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Ensure we can run it standalone
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
