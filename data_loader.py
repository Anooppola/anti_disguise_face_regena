import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class PairedFaceDataset(Dataset):
    """
    Paired dataset loader for masked → unmasked face reconstruction.
    Occluded images: dataset/occluded/M####.png
    Real images:     dataset/real/UM####.png
    Pairing is done by matching the numeric ID (e.g., M0012 ↔ UM0012).
    """

    def __init__(self, root, image_size=256):
        self.occluded_dir = os.path.join(root, "occluded")
        self.real_dir = os.path.join(root, "real")

        if not os.path.isdir(self.occluded_dir):
            raise FileNotFoundError(f"Occluded directory not found: {self.occluded_dir}")
        if not os.path.isdir(self.real_dir):
            raise FileNotFoundError(f"Real directory not found: {self.real_dir}")

        # Build paired list by matching filenames
        occluded_files = sorted(os.listdir(self.occluded_dir))
        real_files_set = set(os.listdir(self.real_dir))

        self.pairs = []
        for occ_name in occluded_files:
            # Check if filenames match directly (same name in both folders)
            if occ_name in real_files_set:
                self.pairs.append((occ_name, occ_name))
            else:
                # Handle M#### → UM#### naming convention
                name, ext = os.path.splitext(occ_name)
                if name.startswith("M"):
                    real_name = "U" + name + ext  # M0012.png → UM0012.png
                    if real_name in real_files_set:
                        self.pairs.append((occ_name, real_name))

        if len(self.pairs) == 0:
            raise ValueError(
                f"No paired images found!\n"
                f"  Occluded dir: {self.occluded_dir} ({len(occluded_files)} files)\n"
                f"  Real dir: {self.real_dir} ({len(real_files_set)} files)\n"
                f"  Make sure filenames match (e.g., M0012.png ↔ UM0012.png or same names)."
            )

        print(f"  Loaded {len(self.pairs)} paired images from '{root}'")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        occ_name, real_name = self.pairs[idx]

        occluded_img = Image.open(
            os.path.join(self.occluded_dir, occ_name)
        ).convert("RGB")

        real_img = Image.open(
            os.path.join(self.real_dir, real_name)
        ).convert("RGB")

        occluded_img = self.transform(occluded_img)
        real_img = self.transform(real_img)

        return {"occluded": occluded_img, "real": real_img}


def get_dataloader(root, batch_size=8, num_workers=0):
    """Create a DataLoader for paired face reconstruction dataset."""
    dataset = PairedFaceDataset(root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )