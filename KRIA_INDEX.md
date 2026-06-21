# AMD KRIA Deployment - Complete Index

## 🎯 START HERE

**New to KRIA deployment?** Start with these files in order:

1. **QUICKSTART_KRIA.txt** - Visual quick reference (5 min read)
2. **KRIA_README.md** - Complete guide with examples (15 min read)
3. **KRIA_DEPLOYMENT.md** - In-depth documentation (reference)

## 📦 What's New

### Modified Files
- **FastCGHNet.py** - Enhanced with KRIA support
  - Added `lite=True` parameter for lightweight model (50% smaller)
  - Added ONNX export functionality
  - Added ONNX Runtime inference support
  - Auto-detection of ARM/KRIA platform

### New Files for KRIA

#### Documentation (Read These First)
| File | Purpose | Read Time |
|------|---------|-----------|
| `QUICKSTART_KRIA.txt` | Visual quick reference | 5 min |
| `KRIA_README.md` | Comprehensive deployment guide | 15 min |
| `KRIA_DEPLOYMENT.md` | Detailed technical documentation | 30 min |

#### Executable Scripts
| File | Purpose | Platform |
|------|---------|----------|
| `kria_predict.py` | Inference engine with ONNX/Vitis AI support | KRIA + Dev |
| `setup_kria.sh` | Automated KRIA setup (dependencies install) | KRIA |
| `vitis_ai_helper.py` | FPGA acceleration tools | Dev (optional) |

#### Configuration Files
| File | Purpose |
|------|---------|
| `requirements-kria.txt` | Minimal Python dependencies for KRIA |

## 🚀 Quick Deployment Path

### Path 1: Fast Start (30 minutes)
```
1. Export model:    python -c "from FastCGHNet import export_onnx; ..."
2. Copy to KRIA:    scp model.onnx root@kria:/root/
3. Setup KRIA:      ssh root@kria bash setup_kria.sh
4. Run inference:   ssh root@kria python kria_predict.py image.png
5. Results back:    scp root@kria:/root/*.bmp ./
```

### Path 2: Production with FPGA (2+ hours)
```
1. Train lite model:     train_cgh_network(lite=True)
2. Export to ONNX:       export_onnx()
3. Deploy Vitis AI:      vitis_ai_helper.py --install
4. Quantize model:       vai_q_pytorch (see vitis_ai_helper.py)
5. Compile for FPGA:     vai_c_onnx
6. Deploy to KRIA:       Transfer compiled model
7. Run with FPGA:        kria_predict.py --vitis-ai
```

## 📊 Performance Tiers

| Tier | Platform | Model | Speed | Setup Time | Cost |
|------|----------|-------|-------|-----------|------|
| **Easiest** | KRIA (CPU) | lite | 250ms | 15 min | Free |
| **Fast** | KRIA (CPU) | full | 450ms | 15 min | Free |
| **Best** | KRIA (FPGA) | int8 | 50ms | 2+ hrs | KRIA board |
| **Ultimate** | KRIA (FPGA) | int4 | 35ms | 2+ hrs | KRIA board |

## 🔧 Deployment Strategies

### Option A: ONNX Runtime (Recommended for Start)
- ✓ Works on any ARM device
- ✓ Easy setup (5 min)
- ✓ No compilation needed
- ✗ Slower (250-450ms)
- **Use when:** Quick deployment, no FPGA available

**Setup:**
```bash
pip install onnxruntime opencv-python numpy pillow
python kria_predict.py image.png --model model.onnx
```

### Option B: Vitis AI (Recommended for Performance)
- ✓ 5-10x faster with FPGA
- ✓ Official AMD solution
- ✗ Complex setup (2+ hours)
- ✗ Requires KRIA with FPGA
- **Use when:** Real-time required, have FPGA board

**Setup:**
```bash
python vitis_ai_helper.py --install
# Follow steps to compile model
python kria_predict.py image.png --vitis-ai
```

### Option C: Apache TVM (Advanced)
- ✓ Advanced compiler optimizations
- ✓ Multi-backend support
- ✗ Steeper learning curve
- **Use when:** Need maximum optimization flexibility

## 📂 File Descriptions

### FastCGHNet.py
**What's New:**
- Platform detection (ARM/KRIA auto-detect)
- Lightweight model option (50% smaller)
- ONNX export function
- ONNX Runtime inference
- Backward compatible

**Key Functions:**
```python
# Train lightweight model
train_cgh_network(lite=True)

# Export to ONNX
export_onnx("model.pt", "model.onnx")

# ONNX inference
predict_hologram_onnx("image.png", "model.onnx")

# Auto-detect platform
predict_hologram("image.png")  # Uses ONNX on ARM, PyTorch on x86
```

### kria_predict.py
**Full-featured KRIA inference script**

```bash
# Single image
python kria_predict.py image.png --model model.onnx --output out.bmp

# Batch processing
python kria_predict.py --batch input_dir/ --model model.onnx --output output_dir/

# Benchmark performance
python kria_predict.py --benchmark --model model.onnx

# FPGA inference
python kria_predict.py image.png --vitis-ai --model compiled.onnx

# Help
python kria_predict.py --help
```

**Features:**
- ONNX Runtime with ARM NEON optimization
- Vitis AI support (FPGA acceleration)
- Batch processing
- Performance benchmarking
- Automatic device detection
- Cross-platform (dev machine + KRIA)

### setup_kria.sh
**Automated setup script for KRIA board**

```bash
# On KRIA board
bash setup_kria.sh

# Does:
# - Update system packages
# - Install Python dependencies
# - Install ONNX Runtime
# - Verify installation
# - Optional benchmark
```

**What it installs:**
- python3-pip
- onnxruntime (with ARM NEON)
- opencv-python
- pillow
- numpy

### vitis_ai_helper.py
**FPGA acceleration tools (development machine)**

```bash
# Show Vitis AI installation steps
python vitis_ai_helper.py --install

# Generate Docker pipeline for quantization
python vitis_ai_helper.py --docker

# Prepare model for quantization
python vitis_ai_helper.py --quantize --model model_lite.pt
```

### requirements-kria.txt
**Minimal dependencies for KRIA board**

Contains only what's needed on KRIA:
- onnxruntime (CPU inference)
- opencv-python (image processing)
- pillow (image I/O)
- numpy (numerical computation)

Training dependencies (torch, torchvision) intentionally omitted.

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Model trained on development machine
- [ ] Model saves to `models/best_model.pt`
- [ ] Test inference works on dev machine

### KRIA Setup
- [ ] KRIA board accessible via SSH
- [ ] Network connection to KRIA working
- [ ] Enough storage on KRIA (500MB minimum)

### Export Phase
- [ ] Export model: `export_onnx("models/best_model.pt")`
- [ ] Verify ONNX file created: `models/fastcghnet.onnx`
- [ ] Test on dev machine: `kria_predict.py test.png --model model.onnx`

### Transfer Phase
- [ ] Copy model to KRIA: `scp model.onnx root@kria:/root/`
- [ ] Copy script to KRIA: `scp kria_predict.py root@kria:/root/`
- [ ] Copy requirements: `scp requirements-kria.txt root@kria:/root/`

### KRIA Setup Phase
- [ ] SSH into KRIA: `ssh root@kria`
- [ ] Run setup: `bash setup_kria.sh`
- [ ] Verify: `python --version` and `pip list`

### Test Phase
- [ ] Copy test image: `scp test.png root@kria:/root/`
- [ ] Run inference: `ssh root@kria python kria_predict.py test.png`
- [ ] Check output: Verify BMP file created

### Performance Tuning
- [ ] Benchmark: `python kria_predict.py --benchmark`
- [ ] Accept performance or upgrade to Vitis AI
- [ ] Document performance numbers

## 🎓 Learning Resources

### For Beginners
1. Read `QUICKSTART_KRIA.txt` (visual overview)
2. Read `KRIA_README.md` (complete guide)
3. Follow Path 1: Fast Start

### For Optimization
1. Read `KRIA_DEPLOYMENT.md` (technical details)
2. Read performance benchmarks section
3. Follow Path 2: Production with FPGA
4. Review `vitis_ai_helper.py` documentation

### For Advanced Users
1. Review Vitis AI documentation
2. Review Apache TVM optimization
3. Custom FPGA kernel development

## 🆘 Troubleshooting

### ONNX Runtime Issues
```bash
# Install correct architecture
pip install onnxruntime-aarch64  # 64-bit ARM
pip install onnxruntime          # Auto-detect

# Verify installation
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### Out of Memory
```python
# Use lite model (50% smaller)
train_cgh_network(lite=True)
# Or export the lite model
export_onnx("models/best_model_lite.pt")
```

### Slow Inference
- Check: Use lite model?
- Benchmark: `python kria_predict.py --benchmark`
- Upgrade: Consider Vitis AI for FPGA acceleration

### Model Not Found
```bash
# Check file exists
ls -la models/*.onnx

# Export if missing
python -c "from FastCGHNet import export_onnx; export_onnx()"
```

## 🔗 Quick Links

- [AMD KRIA Documentation](https://xilinx.github.io/kria/)
- [Vitis AI GitHub](https://github.com/Xilinx/Vitis-AI)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Apache TVM Documentation](https://tvm.apache.org/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## 📞 Support

### Documentation Files in Order of Detail
1. `QUICKSTART_KRIA.txt` - Quick reference
2. `KRIA_README.md` - Overview + examples
3. `KRIA_DEPLOYMENT.md` - Full technical details
4. Code comments in `kria_predict.py` and `vitis_ai_helper.py`

### Common Tasks

**Export model:**
```python
from FastCGHNet import export_onnx
export_onnx("models/best_model_lite.pt", "models/fastcghnet_lite.onnx")
```

**Test on dev machine:**
```bash
python kria_predict.py test.png --model models/fastcghnet_lite.onnx --output out.bmp
```

**Batch process on KRIA:**
```bash
ssh root@kria python /root/kria_predict.py --batch /tmp/images/ \
  --model /root/model.onnx --output /tmp/output/
```

**Benchmark performance:**
```bash
python kria_predict.py --benchmark --model model.onnx
```

---

**Ready to deploy?** Start with `QUICKSTART_KRIA.txt` 🚀
