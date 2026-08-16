import os
import json
from collections import defaultdict

data_dir = "data/raw/PlantVillage"
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

summary = {}
crop_counts = defaultdict(int)
total_images = 0

for category in sorted(categories):
    cat_path = os.path.join(data_dir, category)
    num_files = len([f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))])
    summary[category] = num_files
    total_images += num_files
    
    crop = category.split("_")[0].split("___")[0]
    crop_counts[crop] += num_files

report = {
    "total_images": total_images,
    "num_classes": len(categories),
    "classes": summary,
    "crop_distribution": dict(crop_counts)
}

print("=== DATASET EDA SUMMARY ===")
print(f"Total Images: {total_images}")
print(f"Total Classes: {len(categories)}")
print("\nCrop Distribution:")
for crop, count in crop_counts.items():
    print(f"  - {crop}: {count} images")

os.makedirs("reports", exist_ok=True)
with open("reports/eda_summary.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nEDA report saved to reports/eda_summary.json")
