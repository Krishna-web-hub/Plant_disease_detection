"""End-to-end training pipeline for Plant Disease Detection on NVIDIA RTX 3060.

Runs the complete sequence:
  1. Data verification (auto-runs training/prepare_data.py if manifest is missing)
  2. Crop classifier training (ViT-Base)
  3. Memory cleanup / GPU cache flush
  4. Per-crop disease classifier training (EfficientNet-B0 for Tomato, Potato, Pepper, etc.)
  5. Final status summary report
"""
import argparse
import csv
import json
import os
import sys
import time

import timm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataset import ManifestDataset
from training.engine import fit
from training.prepare_data import main as run_prepare_data
from training.utils import build_transforms, clear_gpu_memory, setup_gpu


def get_available_crops(manifest_path):
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        crops = sorted({r["crop"] for r in reader if r.get("crop")})
    return crops


def train_crop_model(cfg, device, epochs, batch_size, num_workers, lr, use_amp):
    cc = cfg["crop_classifier"]
    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING CROP CLASSIFIER (ViT)")
    print("=" * 60)

    train_tf, eval_tf = build_transforms(cfg["img_size"])
    train_ds = ManifestDataset(cc["manifest"], "train", "crop", transform=train_tf)
    val_ds = ManifestDataset(cc["manifest"], "val", "crop", transform=eval_tf, class_to_idx=train_ds.class_to_idx)

    is_cuda = device.type == "cuda"
    pin_memory = cfg.get("pin_memory", True) and is_cuda
    persistent_workers = (num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )

    with open(cc["class_weights"]) as f:
        weights_dict = json.load(f)
    ordered_classes = sorted(train_ds.class_to_idx, key=train_ds.class_to_idx.get)
    weight_tensor = torch.tensor([weights_dict[c] for c in ordered_classes], dtype=torch.float32).to(device)

    model = timm.create_model(cc["model_name"], pretrained=True, num_classes=len(ordered_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Model: {cc['model_name']} | Classes: {ordered_classes}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Batch: {batch_size}")

    best_acc = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, cfg.get("early_stopping_patience", 5), cc["output_dir"], train_ds.class_to_idx, use_amp=use_amp
    )

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    clear_gpu_memory()
    return best_acc


def train_disease_model(crop, cfg, device, epochs, batch_size, num_workers, lr, use_amp):
    dc = cfg["disease_classifier"]
    print("\n" + "=" * 60)
    print(f"STAGE 2: TRAINING DISEASE CLASSIFIER FOR '{crop}' (EfficientNet)")
    print("=" * 60)

    train_tf, eval_tf = build_transforms(cfg["img_size"])
    train_ds = ManifestDataset(dc["manifest"], "train", "category", crop=crop, transform=train_tf)
    val_ds = ManifestDataset(dc["manifest"], "val", "category", crop=crop, transform=eval_tf,
                              class_to_idx=train_ds.class_to_idx)

    is_cuda = device.type == "cuda"
    pin_memory = cfg.get("pin_memory", True) and is_cuda
    persistent_workers = (num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )

    weights_path = os.path.join(dc["class_weights_dir"], f"{crop}.json")
    with open(weights_path) as f:
        weights_dict = json.load(f)
    ordered_classes = sorted(train_ds.class_to_idx, key=train_ds.class_to_idx.get)
    weight_tensor = torch.tensor([weights_dict[c] for c in ordered_classes], dtype=torch.float32).to(device)

    model = timm.create_model(dc["model_name"], pretrained=True, num_classes=len(ordered_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Model: {dc['model_name']} | Crop: {crop} | Classes ({len(ordered_classes)}): {ordered_classes}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Batch: {batch_size}")

    output_dir = os.path.join(dc["output_dir"], crop)
    best_acc = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, cfg.get("early_stopping_patience", 5), output_dir, train_ds.class_to_idx, use_amp=use_amp
    )

    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    clear_gpu_memory()
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="End-to-End Plant Disease Training Runner (RTX 3060)")
    parser.add_argument("--config", default="config/training_config.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num workers")
    parser.add_argument("--device", default=None, help="Device to use (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--crops", nargs="+", default=None, help="Subset of crops to train, e.g. --crops Tomato Pepper")
    parser.add_argument("--skip-crop", action="store_true", help="Skip training crop classifier")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 mode")
    args = parser.parse_args()

    start_total_time = time.time()
    cfg = yaml.safe_load(open(args.config))

    # Check manifest existence
    manifest_path = cfg["crop_classifier"]["manifest"]
    if not os.path.isfile(manifest_path):
        print(f"Manifest not found at '{manifest_path}'. Running training/prepare_data.py first...")
        run_prepare_data()

    device_name = args.device or cfg.get("device", "cuda")
    use_amp = not args.no_amp and cfg.get("mixed_precision", True)
    enable_tf32 = not args.no_tf32 and cfg.get("tf32", True)
    benchmark = cfg.get("benchmark", True)

    device = setup_gpu(device_name=device_name, enable_tf32=enable_tf32, benchmark=benchmark, seed=cfg.get("seed", 42))

    epochs = args.epochs or cfg.get("epochs", 25)
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    lr = args.lr or cfg.get("lr", 0.0003)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)

    results = {}

    # 1. Train Crop Classifier
    if not args.skip_crop:
        best_crop_acc = train_crop_model(cfg, device, epochs, batch_size, num_workers, lr, use_amp)
        results["Crop Classifier"] = {
            "model": cfg["crop_classifier"]["model_name"],
            "best_val_acc": round(best_crop_acc, 4),
            "output_dir": cfg["crop_classifier"]["output_dir"],
        }
    else:
        print("\n[Skipping Crop Classifier as requested]")

    # 2. Train Disease Classifiers
    available_crops = get_available_crops(manifest_path)
    target_crops = args.crops if args.crops else available_crops

    for crop in target_crops:
        if crop not in available_crops:
            print(f"Warning: Crop '{crop}' not found in manifest ({available_crops}). Skipping.")
            continue
        best_disease_acc = train_disease_model(crop, cfg, device, epochs, batch_size, num_workers, lr, use_amp)
        results[f"Disease Classifier ({crop})"] = {
            "model": cfg["disease_classifier"]["model_name"],
            "best_val_acc": round(best_disease_acc, 4),
            "output_dir": os.path.join(cfg["disease_classifier"]["output_dir"], crop),
        }

    # Summary Report
    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total Time: {total_elapsed / 60:.2f} minutes ({total_elapsed:.1f} seconds)")
    print(f"Device Used: {device}")
    print("-" * 60)
    for model_name, info in results.items():
        print(f"✓ {model_name:30s} | Best Val Acc: {info['best_val_acc']*100:.2f}% | Checkpoint: {info['output_dir']}/best.pt")
    print("=" * 60)
    print("All models are trained and ready for inference with FastAPI backend!")


if __name__ == "__main__":
    main()
