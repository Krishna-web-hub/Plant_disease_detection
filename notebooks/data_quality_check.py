import os
import json
import hashlib
from collections import defaultdict
from PIL import Image

data_dir = "data/raw/PlantVillage"
categories = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))

corrupt_files = []
hashes = defaultdict(list)
class_counts = {}

for category in categories:
    cat_path = os.path.join(data_dir, category)
    files = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
    class_counts[category] = len(files)

    for fname in files:
        fpath = os.path.join(cat_path, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception as e:
            corrupt_files.append({"file": fpath, "error": str(e)})
            continue

        with open(fpath, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes[h].append(fpath)

duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
num_duplicate_files = sum(len(paths) - 1 for paths in duplicates.values())

total = sum(class_counts.values())
max_count = max(class_counts.values())
class_weights = {c: round(max_count / n, 3) for c, n in class_counts.items()}

report = {
    "total_images": total,
    "corrupt_files": corrupt_files,
    "num_corrupt": len(corrupt_files),
    "num_duplicate_groups": len(duplicates),
    "num_duplicate_files": num_duplicate_files,
    "duplicate_groups": {h: paths for h, paths in list(duplicates.items())[:20]},
    "class_counts": class_counts,
    "class_weights": class_weights,
    "imbalance_ratio_max_to_min": round(max_count / min(class_counts.values()), 2)
}

print("=== DATA QUALITY REPORT ===")
print(f"Total images scanned: {total}")
print(f"Corrupt files: {len(corrupt_files)}")
print(f"Duplicate groups: {len(duplicates)} ({num_duplicate_files} redundant files)")
print(f"Class imbalance (max/min): {report['imbalance_ratio_max_to_min']}x")

os.makedirs("reports", exist_ok=True)
with open("reports/data_quality_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved to reports/data_quality_report.json")
