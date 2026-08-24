import argparse
import json
import os

import timm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

try:
    from training.dataset import ManifestDataset
    from training.engine import fit
    from training.utils import build_transforms, setup_gpu
except ImportError:
    from dataset import ManifestDataset
    from engine import fit
    from utils import build_transforms, setup_gpu


def main():
    parser = argparse.ArgumentParser(description="Train Crop-Specific Disease Classifier (EfficientNet)")
    parser.add_argument("--crop", required=True, help="e.g. Tomato, Potato, Pepper")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 for Ampere GPUs")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dc = cfg["disease_classifier"]

    epochs = args.epochs or cfg.get("epochs", 25)
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    lr = args.lr or cfg.get("lr", 0.0003)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    device_name = args.device or cfg.get("device", "cuda")
    use_amp = not args.no_amp and cfg.get("mixed_precision", True)
    enable_tf32 = not args.no_tf32 and cfg.get("tf32", True)
    benchmark = cfg.get("benchmark", True)

    device = setup_gpu(device_name=device_name, enable_tf32=enable_tf32, benchmark=benchmark, seed=cfg.get("seed", 42))
    is_cuda = device.type == "cuda"
    pin_memory = cfg.get("pin_memory", True) and is_cuda
    persistent_workers = (num_workers > 0)

    train_tf, eval_tf = build_transforms(cfg["img_size"])

    train_ds = ManifestDataset(dc["manifest"], "train", "category", crop=args.crop, transform=train_tf)
    val_ds = ManifestDataset(dc["manifest"], "val", "category", crop=args.crop, transform=eval_tf,
                              class_to_idx=train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    weights_path = os.path.join(dc["class_weights_dir"], f"{args.crop}.json")
    with open(weights_path) as f:
        weights_dict = json.load(f)
    ordered_classes = sorted(train_ds.class_to_idx, key=train_ds.class_to_idx.get)
    weight_tensor = torch.tensor([weights_dict[c] for c in ordered_classes], dtype=torch.float32).to(device)

    model = timm.create_model(dc["model_name"], pretrained=True, num_classes=len(ordered_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Disease Classifier ({args.crop}): {dc['model_name']} | Device: {device} | "
          f"Batch Size: {batch_size} | Workers: {num_workers} | Pin Memory: {pin_memory} | "
          f"Classes: {ordered_classes} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    output_dir = os.path.join(dc["output_dir"], args.crop)
    fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, cfg.get("early_stopping_patience", 5), output_dir, train_ds.class_to_idx, use_amp=use_amp)


if __name__ == "__main__":
    main()

