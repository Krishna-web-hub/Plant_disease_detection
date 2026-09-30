import random

import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_gpu(device_name="cuda", enable_tf32=True, benchmark=True, seed=None):
    """Configures PyTorch for NVIDIA GPU (e.g. RTX 3060 Ampere) training.

    Enables TF32 math on Ampere Tensor Cores, cuDNN benchmark mode,
    and sets reproducible seeds if requested.
    """
    if seed is not None:
        set_seed(seed)

    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_name)
        # Enable TensorFloat-32 (TF32) on Ampere (RTX 3060 / sm_86)
        if enable_tf32:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True

        # Benchmark mode autotunes cuDNN convolution kernels
        if benchmark and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

        gpu_name = torch.cuda.get_device_name(device)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"[GPU Setup] Device: {device} ({gpu_name}) | Total VRAM: {total_vram_gb:.2f} GB | TF32: {enable_tf32} | cuDNN Benchmark: {benchmark}")
        return device
    else:
        device = torch.device("cpu")
        print(f"[GPU Setup] CUDA not available or not requested. Using device: {device}")
        return device


def get_gpu_memory_info(device=None):
    """Returns allocated, reserved, and peak VRAM in MB."""
    if not torch.cuda.is_available():
        return None
    dev = device if device is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "max_allocated_mb": round(max_allocated, 1),
    }


def clear_gpu_memory():
    """Flushes PyTorch CUDA memory cache and triggers garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_transforms(img_size, strong_aug=True):
    """Builds training and evaluation transforms.

    When strong_aug=True, applies TrivialAugmentWide, vertical flip, and
    RandomErasing to prevent model overfitting, background memorization,
    and shortcut learning.
    """
    if strong_aug:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class EarlyStopping:
    """Early stops training when monitored metric stops improving.

    Default mode='min' tracks validation loss to avoid saving models whose
    cross-entropy is diverging due to overconfident misclassifications.
    """
    def __init__(self, patience=5, mode="min", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def step(self, value, epoch=0):
        if self.best is None:
            self.best = value
            self.best_epoch = epoch
            return True

        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


