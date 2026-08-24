import json
import os
import time

import torch

try:
    from training.utils import EarlyStopping, clear_gpu_memory, get_gpu_memory_info
except ImportError:
    from utils import EarlyStopping, clear_gpu_memory, get_gpu_memory_info


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, use_amp=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    is_cuda = device.type == "cuda"

    for images, labels in loader:
        # non_blocking=True utilizes pinned host memory for asynchronous DMA transfer
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=(is_cuda and use_amp)):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if is_cuda and use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, patience, output_dir, class_to_idx, use_amp=True):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(is_cuda and use_amp)) if is_cuda else None
    stopper = EarlyStopping(patience=patience, mode="max")

    print(f"Starting training: {epochs} epochs | AMP (Mixed Precision): {use_amp and is_cuda} | Device: {device}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True, use_amp=use_amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - start_time

        mem_str = ""
        if is_cuda:
            mem = get_gpu_memory_info(device)
            if mem:
                mem_str = f" | VRAM: {mem['allocated_mb']}MB (peak {mem['max_allocated_mb']}MB)"

        print(f"epoch {epoch:3d}/{epochs} [{elapsed:.1f}s] | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}{mem_str}")

        improved = stopper.step(val_acc)
        if improved:
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))

        if stopper.should_stop:
            print(f"early stopping at epoch {epoch} (best val_acc {stopper.best:.4f})")
            break

    # Free cache after fit finishes
    clear_gpu_memory()
    return stopper.best

