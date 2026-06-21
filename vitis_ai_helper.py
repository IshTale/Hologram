"""
Vitis AI quantization and deployment helper for FastCGHNet
Requires: Vitis AI SDK installed on KRIA or development machine
"""

import torch
import numpy as np
from pathlib import Path
import os
import sys


def quantize_for_vitis_ai(
    model_path="models/best_model_lite.pt",
    output_dir="vitis_ai_models",
    quant_method="int8",
    calib_data=None,
):
    """
    Prepare model for Vitis AI deployment with quantization
    
    Args:
        model_path: PyTorch model path
        output_dir: Output directory for quantized models
        quant_method: 'int8', 'int4', or 'mixed'
        calib_data: Calibration data for quantization
    
    Requires Vitis AI installed:
        pip install xilinx-vitis-ai
    """
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║ Vitis AI Quantization Helper                          ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    try:
        from pytorch_quantization import quant_modules
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        print("\n⚠️  Vitis AI quantization tools not installed")
        print("\nInstall Vitis AI:")
        print("  pip install xilinx-vitis-ai")
        print("\nOr use ONNX Runtime for CPU inference (no FPGA)")
        return None
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n[INFO] Loading model: {model_path}")
    from FastCGHNet import FastCGHNet
    
    device = torch.device("cpu")
    model = FastCGHNet(lite=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare for quantization
    print(f"[INFO] Preparing quantization ({quant_method})...")
    
    quant_modules.initialize()
    
    # Disable quantization on first and last layers
    model.encoder[0].weight.disable_quantizer = True
    model.decoder[-2].weight.disable_quantizer = True
    
    print(f"[INFO] Quantization setup complete")
    print(f"\nTo deploy on KRIA with Vitis AI:")
    print(f"  1. Export to ONNX: python FastCGHNet.py export {model_path}")
    print(f"  2. Quantize: vai_q_pytorch --input_model model.onnx \\")
    print(f"               --quant_mode {quant_method} \\")
    print(f"               --output_dir {output_dir}/")
    print(f"  3. Compile for board: vai_c_onnx --model model_quantized.onnx \\")
    print(f"                        --arch /opt/vitis_ai/compiler/arch/kria/kv260.json \\")
    print(f"                        --output_dir compiled/")
    
    return model


def install_vitis_ai():
    """Helper to install Vitis AI on development machine"""
    print("\n[INFO] Installing Vitis AI...")
    print("\nOptions:")
    print("  1. Docker (Recommended):")
    print("     docker pull xilinx/vitis-ai:latest")
    print("     docker run -it xilinx/vitis-ai bash")
    print("")
    print("  2. Native (Ubuntu 20.04+):")
    print("     pip install xilinx-vitis-ai")
    print("")
    print("  3. On KRIA board:")
    print("     apt-get install xilinx-vitis-ai-runtime")


def build_docker_pipeline():
    """Generate docker-compose for Vitis AI pipeline"""
    
    dockerfile = """FROM xilinx/vitis-ai:latest

WORKDIR /workspace

RUN pip install opencv-python pillow numpy onnx onnxruntime

COPY FastCGHNet.py .
COPY requirements.txt .
COPY models/ models/

# Quantize model
RUN vai_q_pytorch --input_model models/fastcghnet_lite.onnx \\
                  --quant_mode int8 \\
                  --output_dir quantized/

# Compile for KV260
RUN vai_c_onnx --model quantized/fastcghnet_lite_quantized.onnx \\
               --arch /opt/vitis_ai/compiler/arch/kria/kv260.json \\
               --output_dir compiled/

ENTRYPOINT ["/bin/bash"]
"""
    
    docker_compose = """version: '3'
services:
  vitis-ai:
    image: xilinx/vitis-ai:latest
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: bash
    stdin_open: true
    tty: true
"""
    
    print("\n[INFO] Docker pipeline files:")
    print("\nDockerfile:")
    print(dockerfile)
    print("\ndocker-compose.yml:")
    print(docker_compose)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vitis AI helper for FastCGHNet")
    parser.add_argument("--quantize", action="store_true", help="Quantize model")
    parser.add_argument("--model", default="models/best_model_lite.pt", help="Model path")
    parser.add_argument("--quant-method", default="int8", help="int8, int4, or mixed")
    parser.add_argument("--output-dir", default="vitis_ai_models", help="Output directory")
    parser.add_argument("--install", action="store_true", help="Show installation steps")
    parser.add_argument("--docker", action="store_true", help="Generate Docker pipeline")
    
    args = parser.parse_args()
    
    if args.install:
        install_vitis_ai()
    elif args.docker:
        build_docker_pipeline()
    elif args.quantize:
        quantize_for_vitis_ai(args.model, args.output_dir, args.quant_method)
    else:
        parser.print_help()
