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


def freeze_backbone(model):
    """Freezes all layers except the final classification head."""
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "get_classifier"):
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreezes all parameters for full end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs, patience, output_dir, class_to_idx, use_amp=True, early_stopping_metric="val_loss",
        freeze_epochs=0, resume=False, resume_checkpoint=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(is_cuda and use_amp)) if is_cuda else None

    # Track validation loss (min) or validation accuracy (max)
    mode = "min" if "loss" in early_stopping_metric else "max"
    stopper = EarlyStopping(patience=patience, mode=mode)

    start_epoch = 1
    history = []

    # Checkpoint resumption logic (Security Pillar 2 & 5: Safe state recovery)
    target_ckpt = resume_checkpoint or (os.path.join(output_dir, "checkpoint_last.pt") if resume else None)
    if target_ckpt and os.path.isfile(target_ckpt):
        try:
            print(f"Loading checkpoint from: {target_ckpt}")
            try:
                ckpt = torch.load(target_ckpt, map_location=device, weights_only=True)
            except Exception:
                ckpt = torch.load(target_ckpt, map_location=device)

            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt and optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and scheduler is not None and ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and scaler is not None and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            if "stopper_best" in ckpt:
                stopper.best = ckpt["stopper_best"]
                stopper.best_epoch = ckpt["stopper_best_epoch"]
                stopper.counter = ckpt["stopper_counter"]
            elif "stopper_state" in ckpt:
                s_state = ckpt["stopper_state"]
                stopper.best = s_state.get("best", stopper.best)
                stopper.best_epoch = s_state.get("best_epoch", stopper.best_epoch)
                stopper.counter = s_state.get("counter", stopper.counter)

            history = ckpt.get("history", [])
            last_epoch = ckpt.get("epoch", 0)
            start_epoch = last_epoch + 1

            print(f"✅ Successfully resumed from checkpoint!")
            print(f"   Last epoch completed: {last_epoch} | Resuming at epoch: {start_epoch}/{epochs}")
            print(f"   Best {early_stopping_metric} recorded: {stopper.best:.4f} (at epoch {stopper.best_epoch})")

            if start_epoch > epochs:
                print(f"⚠️ Checkpoint indicates training already completed {last_epoch}/{epochs} epochs.")
                return stopper.best
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint ({e}). Starting fresh training.")
            start_epoch = 1

    if freeze_epochs > 0 and start_epoch <= freeze_epochs:
        freeze_backbone(model)
        print(f"Phase 1: Frozen backbone for first {freeze_epochs} epochs (training classification head only)")
    elif freeze_epochs > 0 and start_epoch > freeze_epochs:
        unfreeze_all(model)
        print(f"Resumed past freeze phase: Backbone is unfrozen for fine-tuning")

    print(f"Starting training: Epochs {start_epoch}->{epochs} | Metric: {early_stopping_metric} ({mode}) | "
          f"AMP: {use_amp and is_cuda} | Device: {device}")

    for epoch in range(start_epoch, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze_all(model)
            print(f"\nPhase 2: Unfreezing all backbone layers for end-to-end fine-tuning (epoch {epoch}/{epochs})")

        start_time = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True, use_amp=use_amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False, use_amp=use_amp)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - start_time

        mem_str = ""
        if is_cuda:
            mem = get_gpu_memory_info(device)
            if mem:
                mem_str = f" | VRAM: {mem['allocated_mb']}MB (peak {mem['max_allocated_mb']}MB)"

        monitor_val = val_loss if early_stopping_metric == "val_loss" else val_acc
        improved = stopper.step(monitor_val, epoch=epoch)

        save_str = " [saved checkpoint]" if improved else ""
        print(f"epoch {epoch:3d}/{epochs} [{elapsed:.1f}s] | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f}{mem_str}{save_str}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "elapsed_seconds": round(elapsed, 2),
        }
        history.append(epoch_record)

        # Full resumable checkpoint state
        full_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "stopper_best": stopper.best,
            "stopper_best_epoch": stopper.best_epoch,
            "stopper_counter": stopper.counter,
            "history": history,
            "class_to_idx": class_to_idx,
        }

        # 1. Save full resumable state
        torch.save(full_checkpoint, os.path.join(output_dir, "checkpoint_last.pt"))

        # 2. If improved, save best deployment weights + best resumable checkpoint
        if improved:
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
            torch.save(full_checkpoint, os.path.join(output_dir, "checkpoint_best.pt"))

        # 3. Save latest deployment weights
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))

        if stopper.should_stop:
            print(f"early stopping triggered at epoch {epoch} (best {early_stopping_metric}: {stopper.best:.4f} at epoch {stopper.best_epoch})")
            break

    # Save training history JSON for loss/ROC analysis
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Free cache after fit finishes
    clear_gpu_memory()
    return stopper.best


