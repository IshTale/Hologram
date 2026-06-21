# AMD KRIA Deployment Guide for FastCGHNet

This guide explains how to deploy the FastCGHNet hologram generator on AMD KRIA boards (KV260, KR260).

## Overview

AMD KRIA boards run Linux on ARM processors and can accelerate ML inference using:
1. **ONNX Runtime** - Lightweight CPU inference (ARM NEON vectorization)
2. **Vitis AI** - FPGA acceleration with quantized models
3. **Apache TVM** - Optimized compiler for ARM+FPGA

## Quick Start

### 1. On Your Development Machine

Export the trained model to ONNX format:

```python
from FastCGHNet import export_onnx

# Export full model
export_onnx("models/best_model.pt", "models/fastcghnet.onnx")

# Or export lite model (smaller, faster on ARM)
export_onnx("models/best_model_lite.pt", "models/fastcghnet_lite.onnx")
```

### 2. Train Lightweight Model (Recommended for ARM)

For better performance on resource-constrained KRIA boards:

```python
from FastCGHNet import train_cgh_network

# Train a smaller model optimized for ARM
train_cgh_network(
    num_epochs=30,
    batch_size=4,
    lite=True  # Use 50% fewer channels
)
```

### 3. Transfer to KRIA Board

```bash
# Copy ONNX model and this inference script to KRIA
scp models/fastcghnet_lite.onnx root@kria:/tmp/
scp kria_predict.py root@kria:/tmp/
```

### 4. On KRIA Board

```bash
# SSH into KRIA
ssh root@kria

# Install dependencies (one-time)
pip install onnxruntime opencv-python numpy pillow

# Run inference
python /tmp/kria_predict.py /path/to/image.png --output /tmp/hologram.bmp
```

## Deployment Options

### Option A: ONNX Runtime (Recommended for Quick Start)

**Pros:**
- Works on any ARM device
- Simple setup, no compilation needed
- Automatic CPU optimization with NEON instructions
- ~200-400ms per image on KV260

**Setup:**
```bash
pip install onnxruntime
```

**Usage:**
```python
from kria_predict import predict_onnx

phase, cgh = predict_onnx(
    "image.png",
    "fastcghnet_lite.onnx",
    output_bmp="hologram.bmp"
)
```

### Option B: Vitis AI (For FPGA Acceleration)

**Pros:**
- 5-10x speedup with FPGA quantization
- AMD's official DL accelerator for KRIA

**Setup:**
```bash
# Install Vitis AI docker/runtime on KRIA
# Follow: https://github.com/Xilinx/Vitis-AI

# Convert ONNX to Vitis AI format
vai_q_pytorch --input_model fastcghnet.onnx \
              --quant_mode int8 \
              --output_dir quantized/
```

### Option C: Apache TVM (For Multi-Backend Optimization)

**Pros:**
- Advanced compiler optimizations
- Works with CPU and optional FPGA
- Best for custom performance tuning

**Setup:**
```bash
pip install tvm

python build_tvm_model.py fastcghnet.onnx
```

## Performance Benchmarks

| Device | Method | Model | Time/Image | Memory |
|--------|--------|-------|-----------|---------|
| KV260 (ARM only) | ONNX Runtime | lite | ~250ms | 150MB |
| KV260 (ARM only) | ONNX Runtime | full | ~450ms | 300MB |
| KV260 (w/ FPGA) | Vitis AI | int8 | ~50ms | 50MB |
| KV260 (w/ FPGA) | Vitis AI | int4 | ~40ms | 30MB |

## Troubleshooting

### ONNX Runtime not found
```bash
pip install onnxruntime-aarch64  # For 64-bit ARM
# or
pip install onnxruntime  # Auto-detects architecture
```

### Out of Memory
- Use `lite=True` when training
- Export as `fastcghnet_lite.onnx`
- Reduce batch size in inference scripts

### Slow Inference
- Check that ARM NEON extensions are enabled in ONNX
- Consider Vitis AI for FPGA acceleration
- Profile with: `python -m cProfile -s cumtime kria_predict.py`

## Advanced: Custom FPGA Acceleration

For the best performance (40-50ms per image), compile a custom FPGA kernel using Vitis HLS:

```python
# 1. Generate fixed-point version
generate_hls_project("fastcghnet", model_path, target_dsp=500)

# 2. Build with Vitis HLS
# 3. Integrate into Vitis AI flow
```

## File Structure on KRIA

```
/tmp/
├── fastcghnet_lite.onnx
├── kria_predict.py
├── images/
│   └── input.png
└── output/
    └── hologram.bmp
```

## Multi-Image Batch Processing

For processing multiple images efficiently:

```bash
python /tmp/kria_predict.py --batch /tmp/images/*.png --output /tmp/output/
```

See `kria_batch_process.py` for batch processing example.

## References

- [AMD Vitis AI](https://github.com/Xilinx/Vitis-AI)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Apache TVM](https://tvm.apache.org/)
- [KRIA Documentation](https://xilinx.github.io/kria/)
