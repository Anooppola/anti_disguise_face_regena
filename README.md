# Zero-Shot Anti-Disguise Facial Reconstruction using GANs

## Project Goal
This deep learning project aims to build a system that reconstructs hidden facial regions when a face is partially occluded by disguise elements (e.g., masks, sunglasses, scarves, or makeup). The system utilizes Generative Adversarial Networks (GANs) combined with image inpainting techniques to predict and reconstruct the missing portions of the face.

## Architecture
- **Generator**: A U-Net based encoder-decoder architecture with skip connections. It ensures the structural integrity of the facial features is preserved from different resolutions.
- **Discriminator**: A PatchGAN-like Convolutional Neural Network (CNN) classifier that distinguishes between real unoccluded faces and the generator's reconstructed ones.
- **Loss Functions**: Adversarial (GAN) Loss combined with L1 Reconstruction Loss for pixel-level accuracy.

## Dataset
You need a clean face dataset. Some potential sources:
- Masked Face Recognition: https://www.kaggle.com/datasets/muhammeddalkran/masked-facerecognition
- CelebA or FFHQ datasets are highly recommended.

Place your dataset images inside the `dataset` folder.

## Setup & Requirements
- Python 3
- PyTorch & Torchvision
- OpenCV
- Matplotlib
- scikit-image (for SSIM metric)

```bash
pip install torch torchvision opencv-python matplotlib scikit-image numpy
```


## How to Train
Run the training script pointing to your dataset folder:

```bash
python train.py --dataset_path path/to/dataset --epochs 50 --batch_size 16
```
This will automatically generate artificial occlusions using `occlusion_generator.py` during training. Models will be saved in `saved_models/`.

## How to Test
Evaluate reconstruction quality using PSNR and SSIM, and visualize the output:

```bash
python test.py --image_path path/to/test_image.jpg --model_path saved_models/generator_X.pth
```
The result will display the Original Image, Disguised Image, and Reconstructed face. Results are also saved as `results/output.png`.

## Ethical Note
**Important:** This system is designed for **research purposes and security enhancement only**. All users must strictly adhere to privacy regulations and laws regarding biometric data and facial generation. Misuse for identity falsification or unauthorized surveillance is strictly condemned.
