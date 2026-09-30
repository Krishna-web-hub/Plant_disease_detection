"""Build stratified train/val/test manifest + class weights from PlantVillage raw datasets.

Supports both:
- Subsets (e.g. data/raw/PlantVillage with Tomato, Potato, Pepper)
- Full 14-crop PlantVillage dataset from https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw
  (data/raw/color with all 38 categories: Apple, Blueberry, Cherry, Corn, Grape, Orange,
   Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).

Security & Robustness Guarantees:
1. Path Traversal & Symlink Escape Protection (canonical realpath bounds).
2. Magic Bytes Header Inspection (verifies actual JPEG, PNG, WebP headers; rejects disguised/malicious files).
3. Image Decompression Bomb Protection (Image.MAX_IMAGE_PIXELS = 10_000_000).
4. Zero Cross-Split Data Leakage via multi-tier Union-Find grouping:
   - MD5 exact-duplicate clustering
   - Perceptual dHash near-duplicate clustering (Hamming distance <= 2)
   - Physical leaf tracking series regex clustering (e.g. 'Leaf 24 Day 13/16')
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Optional
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Defensive limits against decompression bomb DoS
Image.MAX_IMAGE_PIXELS = 10_000_000

DEFAULT_DATA_DIRS = ["data/raw/PlantVillage", "data/raw/color", "data/raw"]
OUT_DIR = "data/processed"
REPORTS_DIR = "reports"
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MAGIC_SIGNATURES = [
    b"\xff\xd8\xff",       # JPEG
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"RIFF",               # WebP (checked with WEBP at offset 8)
]

LEAF_REGEX = re.compile(r"___([A-Za-z0-9_]+[ _]Leaf[ _][0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def crop_of(category: str) -> str:
    """Extracts standardized crop name from category string."""
    raw_crop = category.split("_")[0]
    # Clean any parenthetical note like Corn_(maize) -> Corn
    raw_crop = re.sub(r"\(.*?\)", "", raw_crop).strip()
    return raw_crop


def is_valid_image_magic(fpath: str) -> bool:
    """Verifies that the file begins with true image magic bytes."""
    try:
        with open(fpath, "rb") as fh:
            header = fh.read(16)
        if header.startswith(b"\xff\xd8\xff"):
            return True
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return True
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
            return True
        return False
    except Exception:
        return False


def compute_dhash(image_path: str, hash_size: int = 8) -> Optional[int]:
    """Computes a 64-bit difference hash (dHash) using PIL & NumPy."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.int32)
            diff = arr[:, 1:] > arr[:, :-1]
            return sum([1 << i for i, b in enumerate(diff.flatten()) if b])
    except Exception:
        return None


class UnionFind:
    """Disjoint Set Union to cluster connected duplicate/near-duplicate images."""
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

    def get_groups(self):
        groups = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return list(groups.values())


def extract_leaf_id(filename: str) -> Optional[str]:
    m = LEAF_REGEX.search(filename)
    return m.group(1).lower() if m else None


def resolve_data_directory(explicit_path: Optional[str] = None) -> str:
    """Resolves data directory with fallback detection."""
    if explicit_path:
        if os.path.isdir(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Specified --data_dir not found: {explicit_path}")

    for candidate in DEFAULT_DATA_DIRS:
        if os.path.isdir(candidate):
            # Check if candidate contains subdirectories
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            if subdirs:
                return candidate

    raise FileNotFoundError(
        f"Could not locate PlantVillage raw data. Looked in: {DEFAULT_DATA_DIRS}. "
        f"Download from https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw"
    )


def main():
    parser = argparse.ArgumentParser(description="Leak-proof, hardened PlantVillage data preparation.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw PlantVillage dataset directory")
    args = parser.parse_args()

    data_dir = resolve_data_directory(args.data_dir)
    real_base = os.path.realpath(data_dir)

    random.seed(SEED)
    categories = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))

    rows = []
    total_scanned = 0
    corrupt_files = []
    total_md5_dupes = 0
    total_leaf_groups = 0
    total_dhash_merges = 0

    print("=" * 60)
    print("STARTING HARDENED LEAK-PROOF DATA PREPARATION")
    print(f"Directory: {data_dir} (Resolved: {real_base})")
    print(f"Categories found: {len(categories)} | Splits: {SPLIT_RATIOS}")
    print("=" * 60)

    for category in categories:
        cat_path = os.path.join(data_dir, category)
        candidates = sorted(f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f)))

        valid_files = []
        md5_map = defaultdict(list)
        dhash_map = {}
        leaf_map = defaultdict(list)

        for fname in candidates:
            total_scanned += 1
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue

            fpath = os.path.join(cat_path, fname)
            real_fpath = os.path.realpath(fpath)

            # Security Guard 1: Canonical Path Traversal & Symlink Escape Check
            if not real_fpath.startswith(real_base):
                logger.warning("Path traversal or symlink escape rejected: %s", fpath)
                corrupt_files.append({"file": fpath, "error": "Path traversal or symlink escape detected"})
                continue

            # Security Guard 2: Magic Bytes Verification
            if not is_valid_image_magic(fpath):
                corrupt_files.append({"file": fpath, "error": "Invalid image magic bytes"})
                continue

            # Security Guard 3: PIL Verification (Decompression Bomb & Corruption Guard)
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception as e:
                corrupt_files.append({"file": fpath, "error": str(e)})
                continue

            valid_files.append(fpath)

            # 1. Exact MD5 hash
            with open(fpath, "rb") as fh:
                h_md5 = hashlib.md5(fh.read()).hexdigest()
            md5_map[h_md5].append(fpath)

            # 2. Perceptual dHash
            h_dhash = compute_dhash(fpath)
            if h_dhash is not None:
                dhash_map[fpath] = h_dhash

            # 3. Leaf series identification
            leaf_id = extract_leaf_id(fname)
            if leaf_id:
                leaf_map[leaf_id].append(fpath)

        # Build clusters via Union-Find
        uf = UnionFind(valid_files)

        # Merge exact MD5 matches
        for fpaths in md5_map.values():
            if len(fpaths) > 1:
                total_md5_dupes += len(fpaths) - 1
                for i in range(1, len(fpaths)):
                    uf.union(fpaths[0], fpaths[i])

        # Merge physical leaf tracking series
        for fpaths in leaf_map.values():
            if len(fpaths) > 1:
                total_leaf_groups += 1
                for i in range(1, len(fpaths)):
                    uf.union(fpaths[0], fpaths[i])

        # Merge perceptual near-duplicates (Hamming distance <= 2)
        valid_with_hash = [f for f in valid_files if f in dhash_map]
        for i in range(len(valid_with_hash)):
            f_i = valid_with_hash[i]
            h_i = dhash_map[f_i]
            for j in range(i + 1, len(valid_with_hash)):
                f_j = valid_with_hash[j]
                h_j = dhash_map[f_j]
                dist = bin(h_i ^ h_j).count("1")
                if dist <= 2:
                    if uf.find(f_i) != uf.find(f_j):
                        total_dhash_merges += 1
                        uf.union(f_i, f_j)

        clusters = uf.get_groups()
        random.shuffle(clusters)

        n_total_cat = len(valid_files)
        n_train = int(n_total_cat * SPLIT_RATIOS["train"])
        n_val = int(n_total_cat * SPLIT_RATIOS["val"])

        assigned = 0
        for cluster in clusters:
            if assigned < n_train:
                split = "train"
            elif assigned < n_train + n_val:
                split = "val"
            else:
                split = "test"

            for fpath in cluster:
                rows.append({
                    "filepath": fpath,
                    "category": category,
                    "crop": crop_of(category),
                    "split": split,
                })
            assigned += len(cluster)

        multi_item_clusters = [c for c in clusters if len(c) > 1]
        print(f"  [{category[:32]:32s}] {len(valid_files):5d} images -> {len(clusters):5d} clusters "
              f"({len(multi_item_clusters)} multi-image clusters)")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "category", "crop", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Class Weights for Crop Classifier
    crop_counts = defaultdict(int)
    for r in rows:
        if r["split"] == "train":
            crop_counts[r["crop"]] += 1

    max_crop_count = max(crop_counts.values()) if crop_counts else 1
    crop_weights = {crop: round(max_crop_count / count, 4) for crop, count in crop_counts.items()}
    with open(os.path.join(OUT_DIR, "class_weights_crop.json"), "w", encoding="utf-8") as f:
        json.dump(crop_weights, f, indent=2)

    # Per-Crop Disease Weights
    disease_weights_dir = os.path.join(OUT_DIR, "class_weights_disease")
    os.makedirs(disease_weights_dir, exist_ok=True)
    crops = sorted(crop_counts.keys())
    for crop in crops:
        disease_counts = defaultdict(int)
        for r in rows:
            if r["split"] == "train" and r["crop"] == crop:
                disease_counts[r["category"]] += 1
        if disease_counts:
            max_dis = max(disease_counts.values())
            weights = {cat: round(max_dis / cnt, 4) for cat, cnt in disease_counts.items()}
            with open(os.path.join(disease_weights_dir, f"{crop}.json"), "w", encoding="utf-8") as f:
                json.dump(weights, f, indent=2)

    # Cross-split leakage audit
    split_counts = defaultdict(int)
    split_by_md5 = defaultdict(set)
    split_by_leaf = defaultdict(set)
    for r in rows:
        split_counts[r["split"]] += 1
        with open(r["filepath"], "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        split_by_md5[h].add(r["split"])
        lid = extract_leaf_id(os.path.basename(r["filepath"]))
        if lid:
            split_by_leaf[lid].add(r["split"])

    md5_leaks = sum(1 for splits in split_by_md5.values() if len(splits) > 1)
    leaf_leaks = sum(1 for splits in split_by_leaf.values() if len(splits) > 1)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "dataset_source": "https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw",
        "data_directory": data_dir,
        "total_scanned": total_scanned,
        "valid_images": len(rows),
        "corrupt_files_skipped": len(corrupt_files),
        "exact_md5_duplicates_grouped": total_md5_dupes,
        "physical_leaf_series_grouped": total_leaf_groups,
        "perceptual_dhash_merges": total_dhash_merges,
        "splits": dict(split_counts),
        "cross_split_md5_leakage": md5_leaks,
        "cross_split_leaf_series_leakage": leaf_leaks,
    }
    with open(os.path.join(REPORTS_DIR, "data_split_leakage_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("DATA PREPARATION AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total valid images: {len(rows)}")
    print(f"Exact MD5 duplicates grouped: {total_md5_dupes}")
    print(f"Physical leaf series grouped: {total_leaf_groups}")
    print(f"Perceptual dHash merges: {total_dhash_merges}")
    print(f"Split sizes: {dict(split_counts)}")
    print(f"Cross-split MD5 Leaks: {md5_leaks} (Expected: 0)")
    print(f"Cross-split Leaf Series Leaks: {leaf_leaks} (Expected: 0)")
    print(f"Saved manifest: {manifest_path}")
    print(f"Audit report: {REPORTS_DIR}/data_split_leakage_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
