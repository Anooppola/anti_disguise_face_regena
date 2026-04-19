import os
import io
import sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Ensure 'app' is resolvable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Set an environment variable so the app won't crash if model isn't downloaded during CI
os.environ["MODEL_PATH"] = "dummy_path_for_ci"

from app.main import app

client = TestClient(app)

def test_read_root():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_no_file():
    """Test the predict endpoint without an uploaded file."""
    response = client.post("/predict")
    # FastAPI automatically handles missing body parameters with a 422 Unprocessable Entity
    assert response.status_code == 422

# NOTE: We can only fully test /predict if models exist. In a true CI environment 
# without heavy lifting, we might mock the inference or allow a 500 fallback 
# because of "Model not loaded" error. 
def test_predict_endpoint_with_dummy_image():
    """Test the predict endpoint with a dummy image."""
    
    # Create a dummy 256x256 RGB image in memory
    img = Image.new('RGB', (256, 256), color = 'black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    response = client.post("/predict", files={"file": ("dummy.png", buf, "image/png")})
    
    # If the model is not found (which happens in pure dummy CI environment), 
    # we expect our explicit 500 error from app/main.py.
    # If the model IS found, it will return 200.
    assert response.status_code in [200, 500]
    
    if response.status_code == 500:
        assert "Model not loaded" in response.json()["detail"]
    else:
        assert response.headers["content-type"] == "image/png"
