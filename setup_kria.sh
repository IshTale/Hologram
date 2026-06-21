#!/bin/bash
# Quick setup script for KRIA deployment
# Run on KRIA board: bash setup_kria.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  FastCGHNet - AMD KRIA Setup                              ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Check architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [[ ! "$ARCH" =~ armv7|aarch64 ]]; then
    echo "⚠️  This script is for ARM architecture"
    echo "   Current: $ARCH"
    exit 1
fi

# Detect if running on actual KRIA
if [ -f /proc/device-tree/model ]; then
    BOARD=$(cat /proc/device-tree/model)
    echo "✓ Detected board: $BOARD"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 1: Update system packages"
echo "═══════════════════════════════════════════════════════════════"

apt-get update -qq
apt-get install -y python3-pip python3-dev

echo "✓ System packages installed"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 2: Install Python dependencies"
echo "═══════════════════════════════════════════════════════════════"

# Use pre-built wheels for ARM when available
pip install --upgrade pip setuptools wheel -q

echo "Installing ONNX Runtime for ARM..."
pip install onnxruntime -q

echo "Installing image processing libraries..."
pip install opencv-python pillow numpy -q

echo "✓ Python dependencies installed"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 3: Verify installation"
echo "═══════════════════════════════════════════════════════════════"

python3 -c "import onnxruntime; print(f'✓ ONNX Runtime version: {onnxruntime.__version__}')"
python3 -c "import cv2; print(f'✓ OpenCV version: {cv2.__version__}')"
python3 -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 4: Benchmark (optional)"
echo "═══════════════════════════════════════════════════════════════"

read -p "Run benchmark? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "fastcghnet_lite.onnx" ]; then
        echo "Benchmarking FastCGHNet..."
        python3 kria_predict.py --benchmark --model fastcghnet_lite.onnx
    else
        echo "⚠️  Model not found. Copy .onnx files first"
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete!                                          ║"
echo "╚════════════════════════════════════════════════════════════╝"

echo ""
echo "Next steps:"
echo "  1. Copy model files:"
echo "     scp models/fastcghnet_lite.onnx root@kria:/root/"
echo ""
echo "  2. Copy inference script:"
echo "     scp kria_predict.py root@kria:/root/"
echo ""
echo "  3. Run inference:"
echo "     python3 /root/kria_predict.py image.png --output hologram.bmp"
echo ""

echo "Documentation: See KRIA_DEPLOYMENT.md"
