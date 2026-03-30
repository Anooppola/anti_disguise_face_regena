import argparse
from fileinput import filename
import os
import sys
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

# Fix import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import glob

from models.generator import Generator


'''def test_single(model, image_path, transform, device, output_dir):
    """Test on a single masked image and save the result."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Generate unmasked face
    with torch.no_grad():
        output = model(input_tensor)

    # Save output
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_unmasked.png")

    # Save comparison: input | generated
    comparison = torch.cat([input_tensor, output], dim=0)
    vutils.save_image(
        comparison,
        os.path.join(output_dir, f"{basename}_comparison.png"),
        nrow=2,
        normalize=True,
        value_range=(-1, 1)
    )

    # Save generated image only
    vutils.save_image(
        output,
        output_path,
        normalize=True,
        value_range=(-1, 1)
    )

    print(f"  ✅ Saved: {output_path}")
    return output_path  '''

def test_single(model, image_path, transform, device, output_dir):
    """Test single image + calculate PSNR & SSIM"""

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Load input
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Generate output
    with torch.no_grad():
        output = model(input_tensor)

    # Save output
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_unmasked.png")

    vutils.save_image(
        output,
        output_path,
        normalize=True,
        value_range=(-1, 1)
    )

    # ===============================
    # 🔥 PSNR & SSIM CALCULATION
    # ===============================

    # Ground truth path
    filename = os.path.basename(image_path)
    gt_path = os.path.join("dataset/real", filename)

    if os.path.exists(gt_path):

        real = cv2.imread(gt_path)
        generated = cv2.imread(output_path)

        real = cv2.resize(real, (256, 256))
        generated = cv2.resize(generated, (256, 256))

        # PSNR
        psnr_value = peak_signal_noise_ratio(real, generated)

        # SSIM (grayscale)
        real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

        ssim_value = structural_similarity(real_gray, gen_gray)

        print(f"  📊 PSNR: {psnr_value:.2f}")
        print(f"  📊 SSIM: {ssim_value:.4f}")

    else:
        print("  ⚠️ Ground truth not found → skipping metrics")

    print(f"  ✅ Saved: {output_path}")

    return output_path


def test():
    parser = argparse.ArgumentParser(description="Pix2Pix GAN Testing - Generate Unmasked Faces")

    parser.add_argument('--image', type=str, default=None,
                        help='path to a single masked image to test')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='directory of masked images to test (batch mode)')
    parser.add_argument('--model_path', type=str, default='saved_models/generator_best.pth',
                        help='path to trained generator model')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='directory to save results')

    opt = parser.parse_args()

    # Validate inputs
    if opt.image is None and opt.input_dir is None:
        print("⚠️  No input specified. Using default: dataset/occluded/")
        opt.input_dir = "dataset/occluded"

    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # ============================================================
    # Device Setup
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*50}")
    print(f"  Pix2Pix GAN Testing")
    print(f"  Device: {device}")
    print(f"{'='*50}")

    # ============================================================
    # Load Model
    # ============================================================
    if not os.path.exists(opt.model_path):
        print(f"❌ Model not found: {opt.model_path}")
        print(f"   Available models:")
        if os.path.isdir("saved_models"):
            for f in os.listdir("saved_models"):
                if f.endswith(".pth"):
                    print(f"     - saved_models/{f}")
        else:
            print("     No saved_models/ directory found.")
        print(f"\n   Train first: python train.py")
        return

    model = Generator(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  ✅ Model loaded: {opt.model_path}")

    # ============================================================
    # Transform (must match training)
    # ============================================================
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
    ])

    # ============================================================
    # Run Inference
    # ============================================================
    if opt.image:
        # Single image mode
        print(f"\n  Testing single image: {opt.image}")
        test_single(model, opt.image, transform, device, opt.output_dir)

    elif opt.input_dir:
        # Batch mode: process all images in directory
        if not os.path.isdir(opt.input_dir):
            print(f"❌ Directory not found: {opt.input_dir}")
            return

        image_paths = sorted(
            glob.glob(os.path.join(opt.input_dir, "*.png")) +
            glob.glob(os.path.join(opt.input_dir, "*.jpg")) +
            glob.glob(os.path.join(opt.input_dir, "*.jpeg"))
        )

        if len(image_paths) == 0:
            print(f"❌ No images found in: {opt.input_dir}")
            return

        print(f"\n  Processing {len(image_paths)} images from: {opt.input_dir}\n")

        for idx, img_path in enumerate(image_paths):
            test_single(model, img_path, transform, device, opt.output_dir)

            if (idx + 1) % 50 == 0:
                print(f"  ... processed {idx+1}/{len(image_paths)}")

        print(f"\n{'='*50}")
        print(f"  ✅ Testing Complete!")
        print(f"  Processed: {len(image_paths)} images")
        print(f"  Results saved to: {opt.output_dir}/")
        print(f"{'='*50}")


if __name__ == "__main__":
    test()