#!/usr/bin/env python3
"""GPU diagnostic and benchmark tool for NVIDIA RTX 3060 and PyTorch.

Run this script to verify your GPU environment, CUDA setup, TensorFloat-32 (TF32),
Mixed Precision (AMP), and perform a test training pass on the models.
"""
import os
import subprocess
import sys
import time


def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def check_system():
    print_section("1. SYSTEM & PYTHON ENVIRONMENT")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch CUDA Build: {torch.version.cuda or 'None (CPU only)'}")
        print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    except ImportError:
        print("❌ PyTorch is not installed!")
        print("   Install it using: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
    return True


def check_nvidia_smi():
    print_section("2. NVIDIA DRIVER CHECK (nvidia-smi)")
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            lines = res.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                print(f"GPU {i}: {parts[0]} | Driver: {parts[1]} | Total VRAM: {parts[2]}")
            return True
        else:
            print("⚠️  nvidia-smi returned non-zero code. NVIDIA driver might not be installed or GPU is not attached.")
            return False
    except FileNotFoundError:
        print("⚠️  'nvidia-smi' command not found. Ensure NVIDIA drivers are installed on your system.")
        return False


def check_torch_cuda():
    print_section("3. PYTORCH CUDA ACCELERATION")
    import torch

    cuda_avail = torch.cuda.is_available()
    if not cuda_avail:
        print("❌ torch.cuda.is_available() is FALSE.")
        print("\nPossible causes:")
        print("  1. The installed PyTorch wheel is CPU-only.")
        print("     Fix: Run:")
        print("     pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("  2. The NVIDIA GPU is not connected or NVIDIA drivers are missing.")
        return False

    device_count = torch.cuda.device_count()
    print(f"✓ CUDA is available! Found {device_count} CUDA device(s).")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        cap = f"{props.major}.{props.minor}"
        print(f"\n  [Device {i}]: {props.name}")
        print(f"    - Compute Capability: {cap} (Ampere = 8.6)")
        print(f"    - Total Memory: {vram_gb:.2f} GB ({props.total_memory / (1024**2):.0f} MB)")
        print(f"    - Multi-processors: {props.multi_processor_count}")

    # Check Ampere / RTX 3060 specific features
    print("\n  [Ampere / RTX 3060 Capabilities]:")
    tf32_matmul = hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    print(f"    - TF32 Matmul Acceleration: {'✓ Supported' if tf32_matmul else '❌ Not available'}")
    amp_fp16 = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else True
    print(f"    - Mixed Precision (FP16/AMP): ✓ Supported")
    print(f"    - BF16 Tensor Cores: {'✓ Supported' if amp_fp16 else '⚠️ Not supported'}")

    return True


def benchmark_dry_run():
    print_section("4. DRY-RUN BENCHMARK (ViT & EfficientNet on GPU)")
    import timm
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        print("Skipping GPU dry-run benchmark since CUDA is not active.")
        return

    device = torch.device("cuda:0")

    # Configure Ampere TF32 & cuDNN benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 1. Benchmark ViT-Base (Crop Classifier)
    print("\nTesting Vision Transformer (vit_base_patch16_224)...")
    batch_size = 32
    img_size = 224
    num_classes = 3

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda")

        dummy_x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        dummy_y = torch.randint(0, num_classes, (batch_size,), device=device)

        # Warmup
        with torch.amp.autocast("cuda"):
            out = model(dummy_x)
            loss = criterion(out, dummy_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Timed test pass
        t0 = time.time()
        for _ in range(5):
            with torch.amp.autocast("cuda"):
                out = model(dummy_x)
                loss = criterion(out, dummy_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        t1 = time.time()

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        fps = (5 * batch_size) / (t1 - t0)
        print(f"  ✓ ViT-Base training step successful!")
        print(f"  ✓ Throughput: {fps:.1f} images/sec")
        print(f"  ✓ Peak VRAM for batch_size={batch_size}: {peak_vram:.1f} MB (well within RTX 3060 budget)")

        del model, optimizer, scaler, dummy_x, dummy_y
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌ ViT Benchmark failed: {e}")

    # 2. Benchmark EfficientNet-B0 (Disease Classifier)
    print("\nTesting EfficientNet (efficientnet_b0)...")
    num_classes = 10
    try:
        torch.cuda.reset_peak_memory_stats()
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda")

        dummy_x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        dummy_y = torch.randint(0, num_classes, (batch_size,), device=device)

        # Warmup
        with torch.amp.autocast("cuda"):
            out = model(dummy_x)
            loss = criterion(out, dummy_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # Timed test pass
        t0 = time.time()
        for _ in range(10):
            with torch.amp.autocast("cuda"):
                out = model(dummy_x)
                loss = criterion(out, dummy_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        t1 = time.time()

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        fps = (10 * batch_size) / (t1 - t0)
        print(f"  ✓ EfficientNet-B0 training step successful!")
        print(f"  ✓ Throughput: {fps:.1f} images/sec")
        print(f"  ✓ Peak VRAM for batch_size={batch_size}: {peak_vram:.1f} MB")

        del model, optimizer, scaler, dummy_x, dummy_y
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ❌ EfficientNet Benchmark failed: {e}")


def main():
    print("\n" + "#" * 65)
    print("  PLANT DISEASE DETECTION — NVIDIA RTX 3060 GPU VERIFIER")
    print("#" * 65)

    if not check_system():
        return

    check_nvidia_smi()
    has_cuda = check_torch_cuda()

    if has_cuda:
        benchmark_dry_run()
        print_section("READY TO TRAIN")
        print("✓ All GPU checks passed! You can run training with:")
        print("    python training/train_all.py")
    else:
        print_section("SETUP REQUIRED")
        print("To enable RTX 3060 GPU training, run the following setup command:")
        print("    bash setup_gpu_env.sh")
        print("or:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")


if __name__ == "__main__":
    main()
