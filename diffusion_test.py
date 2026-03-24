"""
Diffusion-based face inpainting using Stable Diffusion.
NOTE: This is an ALTERNATIVE approach. The main project uses Pix2Pix GAN.
      This file requires additional dependencies: diffusers, cv2
      Install them with: pip install diffusers opencv-python transformers accelerate
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
torch.set_float32_matmul_precision('high')

from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV is required. Install with: pip install opencv-python")

try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    raise ImportError("diffusers is required. Install with: pip install diffusers transformers accelerate")


def run_diffusion_inpainting(input_path="input.png", output_path="output.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # Load image
    image = Image.open(input_path).convert("RGB").resize((512, 512))

    # Create mask (white = remove mask area)
    img_np = np.array(image)
    mask = cv2.inRange(img_np, (200, 200, 200), (255, 255, 255))
    mask = cv2.medianBlur(mask, 15)
    mask = Image.fromarray(mask).convert("RGB").resize((512, 512))

    # Prompts
    prompt = "realistic human face, natural skin, high detail, 8k portrait, sharp focus"
    negative_prompt = "blurry, low quality, distorted face, bad anatomy"

    # Generate output
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=8,
        num_inference_steps=50
    ).images[0]

    # Save result
    result.save(output_path)
    print(f"✅ Output saved as {output_path}")


if __name__ == "__main__":
    run_diffusion_inpainting()