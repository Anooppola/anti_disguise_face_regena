"""
data_loader.py - Dataset loading and preprocessing for Anti-Disguise GAN
Expects paired images:  masked/M<id>.png  ↔  unmasked/UM<id>.png
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Default transforms ────────────────────────────────────────────────────────

def get_transforms(image_size: int = 256, augment: bool = False):
    """Return torchvision transform pipeline."""
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
    ]
    return transforms.Compose(ops)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MaskedFaceDataset(Dataset):
    """
    Paired dataset.
    masked_dir   : contains files like M0001.png
    unmasked_dir : contains files like UM0001.png
    Pairs are matched by the numeric ID in the filename.
    """

    def __init__(
        self,
        masked_dir: str,
        unmasked_dir: str,
        image_size: int = 256,
        augment: bool = False,
    ):
        self.masked_dir   = Path(masked_dir)
        self.unmasked_dir = Path(unmasked_dir)
        self.transform    = get_transforms(image_size, augment)

        self.pairs = self._build_pairs()
        logger.info("Dataset: %d valid pairs found", len(self.pairs))

    def _parse_id(self, filename: str) -> Optional[str]:
        """Extract numeric ID from filename like M0001.png or UM0001.png."""
        stem = Path(filename).stem          # e.g. "M0001" or "UM0001"
        digits = "".join(filter(str.isdigit, stem))
        return digits if digits else None

    def _build_pairs(self):
        masked_files   = {self._parse_id(f): f for f in os.listdir(self.masked_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))}
        unmasked_files = {self._parse_id(f): f for f in os.listdir(self.unmasked_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg"))}

        common_ids = set(masked_files.keys()) & set(unmasked_files.keys())
        pairs = []
        for uid in sorted(common_ids):
            pairs.append((
                self.masked_dir   / masked_files[uid],
                self.unmasked_dir / unmasked_files[uid],
            ))

        if not pairs:
            logger.warning(
                "No paired images found. Check that masked/ and unmasked/ "
                "directories contain files with matching numeric IDs."
            )
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_path, unmasked_path = self.pairs[idx]
        masked   = Image.open(masked_path).convert("RGB")
        unmasked = Image.open(unmasked_path).convert("RGB")
        return self.transform(masked), self.transform(unmasked)


# ─── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    masked_dir: str,
    unmasked_dir: str,
    image_size: int = 256,
    batch_size: int = 4,
    val_split: float = 0.1,
    num_workers: int = 0,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).
    val_split: fraction of data reserved for validation (0.0–1.0)
    """
    full_dataset = MaskedFaceDataset(masked_dir, unmasked_dir, image_size, augment=False)

    val_size   = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply augmentation only to training split
    if augment_train:
        train_ds.dataset.transform = get_transforms(image_size, augment=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))
    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke-test (adjust paths as needed)
    from pathlib import Path
    base = Path(__file__).parent.parent / "data"
    train_l, val_l = get_dataloaders(
        str(base / "masked"), str(base / "unmasked"), batch_size=2
    )
    masked_batch, unmasked_batch = next(iter(train_l))
    print(f"Masked batch:   {masked_batch.shape}")
    print(f"Unmasked batch: {unmasked_batch.shape}")
