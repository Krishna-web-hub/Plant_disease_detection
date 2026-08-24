#!/usr/bin/env bash
# Environment setup script for NVIDIA GeForce RTX 3060 (Ampere sm_86)
# Installs PyTorch with CUDA 12.1 support and all required dependencies.

set -e

echo "============================================================"
echo " Setting up Plant Disease Detection GPU Environment (RTX 3060)"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.10+."
    exit 1
fi

echo "1. Upgrading pip and build tools..."
python3 -m pip install --upgrade pip setuptools wheel

echo "2. Installing PyTorch & Torchvision with CUDA 12.1 support..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "3. Installing project dependencies..."
python3 -m pip install -r requirements.txt

echo "4. Verifying GPU setup..."
python3 check_gpu.py

echo "============================================================"
echo " Setup complete! Run training with: python training/train_all.py"
echo "============================================================"
