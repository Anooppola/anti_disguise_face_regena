"""
inference.py - Load trained Generator and run prediction on a single image
"""

import io
import logging
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src.model import Generator
from src.preprocessing import preprocess_image, postprocess_tensor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Wraps the Generator model for single-image inference.
    Thread-safe for FastAPI (model loaded once at startup).
    """

    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.model = self._load_model(str(model_path))
        logger.info("InferenceEngine ready on device: %s", self.device)

    def _load_model(self, model_path: str) -> Generator:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = Generator(in_channels=3, out_channels=3).to(self.device)

        state = torch.load(model_path, map_location=self.device, weights_only=True)
        # Support both raw state_dicts and checkpoint dicts
        if isinstance(state, dict) and "generator" in state:
            state = state["generator"]
        model.load_state_dict(state)
        model.eval()

        logger.info("Model loaded from: %s", model_path)
        return model

    @torch.no_grad()
    def predict(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """
        Run inference on a masked face image.
        Accepts: PIL Image, file path, or raw bytes.
        Returns: PIL Image (reconstructed face)
        """
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")

        input_tensor = preprocess_image(image).to(self.device)        # (1,3,256,256)
        output_tensor = self.model(input_tensor)                       # (1,3,256,256)
        return postprocess_tensor(output_tensor)

    @torch.no_grad()
    def predict_bytes(self, image_bytes: bytes, fmt: str = "PNG") -> bytes:
        """
        Convenience method: accept raw bytes, return PNG bytes.
        """
        result_img = self.predict(image_bytes)
        buf = io.BytesIO()
        result_img.save(buf, format=fmt)
        return buf.getvalue()


# ─── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run GAN inference on a single image")
    parser.add_argument("--model",  required=True, help="Path to generator .pth file")
    parser.add_argument("--input",  required=True, help="Path to masked input image")
    parser.add_argument("--output", default="output.png", help="Output image path")
    args = parser.parse_args()

    engine = InferenceEngine(args.model)
    result = engine.predict(args.input)
    result.save(args.output)
    print(f"Result saved to: {args.output}")
