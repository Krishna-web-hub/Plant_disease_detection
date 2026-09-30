import csv

from PIL import Image
from torch.utils.data import Dataset

# Prevent PIL decompression bomb DoS attacks (Security Pillar 3)
Image.MAX_IMAGE_PIXELS = 10_000_000


class ManifestDataset(Dataset):
    """Reads data/processed/manifest.csv and serves (image, label) pairs.

    class_to_idx is built from the train split and must be passed explicitly
    when constructing val/test datasets, so label indices stay consistent
    across splits.
    """

    def __init__(self, manifest_path, split, label_col, crop=None, transform=None, class_to_idx=None):
        self.transform = transform

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["split"] == split and (crop is None or r["crop"] == crop)]

        if class_to_idx is None:
            labels = sorted({r[label_col] for r in rows})
            class_to_idx = {c: i for i, c in enumerate(labels)}
        self.class_to_idx = class_to_idx
        self.samples = [(r["filepath"], class_to_idx[r[label_col]]) for r in rows]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Defensive fallback for corrupt or non-image files (Security Pillar 3)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label
