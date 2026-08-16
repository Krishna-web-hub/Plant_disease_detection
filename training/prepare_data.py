"""Build stratified train/val/test manifest + class weights from data/raw/PlantVillage.

Splits are done per-category (crop+disease folder) so every split gets a
proportional slice of every class. Exact-duplicate images (same MD5) are
kept together in one split so no image leaks between train and test.
Run once before training; re-run if new crops/images are added to data/raw.
"""
import csv
import hashlib
import json
import os
import random
from collections import defaultdict

DATA_DIR = "data/raw/PlantVillage"
OUT_DIR = "data/processed"
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42


def crop_of(category: str) -> str:
    return category.split("_")[0]


def main():
    random.seed(SEED)
    categories = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))

    rows = []
    for category in categories:
        cat_path = os.path.join(DATA_DIR, category)
        files = sorted(f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f)))

        # Group exact-duplicate content so copies never split across train/val/test.
        groups = defaultdict(list)
        for fname in files:
            fpath = os.path.join(cat_path, fname)
            with open(fpath, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            groups[h].append(fpath)

        group_list = list(groups.values())
        random.shuffle(group_list)

        n = len(files)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        assigned = 0
        for group in group_list:
            split = "train" if assigned < n_train else "val" if assigned < n_train + n_val else "test"
            for fpath in group:
                rows.append({
                    "filepath": fpath,
                    "category": category,
                    "crop": crop_of(category),
                    "split": split,
                })
            assigned += len(group)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "category", "crop", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # --- crop-level class weights (train split only) ---
    crop_train_counts = defaultdict(int)
    for r in rows:
        if r["split"] == "train":
            crop_train_counts[r["crop"]] += 1
    max_crop = max(crop_train_counts.values())
    crop_weights = {c: round(max_crop / n, 4) for c, n in crop_train_counts.items()}
    with open(os.path.join(OUT_DIR, "class_weights_crop.json"), "w") as f:
        json.dump(crop_weights, f, indent=2)

    # --- per-crop disease-level class weights (train split only) ---
    disease_dir = os.path.join(OUT_DIR, "class_weights_disease")
    os.makedirs(disease_dir, exist_ok=True)
    per_crop_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["split"] == "train":
            per_crop_counts[r["crop"]][r["category"]] += 1

    for crop, counts in per_crop_counts.items():
        max_c = max(counts.values())
        weights = {cat: round(max_c / n, 4) for cat, n in counts.items()}
        with open(os.path.join(disease_dir, f"{crop}.json"), "w") as f:
            json.dump(weights, f, indent=2)

    # --- summary ---
    split_counts = defaultdict(int)
    for r in rows:
        split_counts[r["split"]] += 1

    print("=== DATA PREP SUMMARY ===")
    print(f"Total images: {len(rows)}")
    print(f"Crops: {sorted(crop_train_counts.keys())}")
    print(f"Split sizes: {dict(split_counts)}")
    print(f"Manifest: {manifest_path}")
    print(f"Crop class weights: {OUT_DIR}/class_weights_crop.json")
    print(f"Disease class weights: {disease_dir}/<crop>.json")


if __name__ == "__main__":
    main()
