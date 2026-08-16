import argparse
import json
import os

import timm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import ManifestDataset
from engine import fit
from utils import build_transforms, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", required=True, help="e.g. Tomato, Potato, Pepper")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dc = cfg["disease_classifier"]
    epochs = args.epochs or cfg["epochs"]
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = build_transforms(cfg["img_size"])

    train_ds = ManifestDataset(dc["manifest"], "train", "category", crop=args.crop, transform=train_tf)
    val_ds = ManifestDataset(dc["manifest"], "val", "category", crop=args.crop, transform=eval_tf,
                              class_to_idx=train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    weights_path = os.path.join(dc["class_weights_dir"], f"{args.crop}.json")
    with open(weights_path) as f:
        weights_dict = json.load(f)
    ordered_classes = sorted(train_ds.class_to_idx, key=train_ds.class_to_idx.get)
    weight_tensor = torch.tensor([weights_dict[c] for c in ordered_classes], dtype=torch.float32).to(device)

    model = timm.create_model(dc["model_name"], pretrained=True, num_classes=len(ordered_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"device: {device} | crop: {args.crop} | classes: {ordered_classes} "
          f"| train: {len(train_ds)} | val: {len(val_ds)}")

    output_dir = os.path.join(dc["output_dir"], args.crop)
    fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, cfg["early_stopping_patience"], output_dir, train_ds.class_to_idx)


if __name__ == "__main__":
    main()
