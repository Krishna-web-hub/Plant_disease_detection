import json
import os

import torch

from utils import EarlyStopping


def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, output_dir, class_to_idx):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    stopper = EarlyStopping(patience=patience, mode="max")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step()

        print(f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        improved = stopper.step(val_acc)
        if improved:
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))

        if stopper.should_stop:
            print(f"early stopping at epoch {epoch} (best val_acc {stopper.best:.4f})")
            break

    return stopper.best
