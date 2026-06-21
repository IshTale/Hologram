# FastCGHNet KRIA Deployment Checklist

Use this checklist to guide you through deploying FastCGHNet to AMD KRIA.

## Phase 1: Preparation (Development Machine)

### Prerequisites
- [ ] FastCGHNet model trained and saved
- [ ] Model file: `models/best_model.pt` exists
- [ ] PyTorch, NumPy, OpenCV installed on dev machine
- [ ] SSH access to KRIA board
- [ ] Network connectivity to KRIA

### Model Export
- [ ] Read `QUICKSTART_KRIA.txt` (visual guide)
- [ ] Read `KRIA_README.md` (complete guide)
- [ ] Export model to ONNX:
  ```python
  from FastCGHNet import export_onnx
  export_onnx("models/best_model_lite.pt", "models/fastcghnet_lite.onnx")
  ```
- [ ] Verify ONNX file created: `models/fastcghnet_lite.onnx`
- [ ] Test inference on dev machine:
  ```bash
  python kria_predict.py test_image.png --model models/fastcghnet_lite.onnx
  ```
- [ ] Output looks correct (BMP file generated)

### File Preparation
- [ ] Gather all files needed:
  - `models/fastcghnet_lite.onnx`
  - `kria_predict.py`
  - `requirements-kria.txt`
  - Any test images
- [ ] Create transfer directory: `mkdir kria_deploy`
- [ ] Copy files to transfer directory

## Phase 2: Transfer (Development Machine → KRIA)

### SSH Configuration
- [ ] KRIA board is powered on and connected
- [ ] Can ping KRIA: `ping <kria-ip>`
- [ ] Can SSH to KRIA: `ssh root@<kria-ip>`
- [ ] Have root password or SSH key configured

### File Transfer
- [ ] Transfer ONNX model:
  ```bash
  scp models/fastcghnet_lite.onnx root@<kria-ip>:/root/
  ```
  - [ ] Verify: `ssh root@<kria-ip> ls -lh /root/fastcghnet_lite.onnx`

- [ ] Transfer inference script:
  ```bash
  scp kria_predict.py root@<kria-ip>:/root/
  ```
  - [ ] Verify: `ssh root@<kria-ip> ls -lh /root/kria_predict.py`

- [ ] Transfer requirements:
  ```bash
  scp requirements-kria.txt root@<kria-ip>:/root/
  ```

- [ ] Transfer test image(s):
  ```bash
  scp test_image.png root@<kria-ip>:/root/
  ```

## Phase 3: KRIA Setup (First Time Only)

### Initial Setup
- [ ] SSH into KRIA:
  ```bash
  ssh root@<kria-ip>
  ```

- [ ] Verify basic system:
  ```bash
  uname -m          # Should show aarch64 or armv7l
  python3 --version # Should show Python 3.x
  df -h /           # Check available disk space
  ```

- [ ] Run automated setup (RECOMMENDED):
  ```bash
  bash /root/setup_kria.sh
  ```
  - [ ] System packages updated
  - [ ] Python dependencies installed
  - [ ] ONNX Runtime installed
  - [ ] All verifications passed

### Manual Setup (Alternative)
If automated setup fails:
- [ ] Update system:
  ```bash
  apt-get update && apt-get upgrade -y
  ```
- [ ] Install dependencies:
  ```bash
  pip install -r /root/requirements-kria.txt
  ```
- [ ] Verify ONNX Runtime:
  ```bash
  python3 -c "import onnxruntime; print(onnxruntime.__version__)"
  ```

### Disk Space Check
- [ ] Available disk space: `df -h /`
- [ ] Minimum required: 500MB
- [ ] Recommended: 1GB or more

## Phase 4: Testing (KRIA Board)

### Single Image Test
- [ ] Run inference on test image:
  ```bash
  cd /root
  python3 kria_predict.py test_image.png \
    --model fastcghnet_lite.onnx \
    --output test_hologram.bmp
  ```
- [ ] Check output file created: `ls -lh test_hologram.bmp`
- [ ] Inference completed successfully
- [ ] Performance acceptable (check output for timing)

### Verify Output
- [ ] Transfer result back to dev machine:
  ```bash
  scp root@<kria-ip>:/root/test_hologram.bmp ./
  ```
- [ ] View output image
- [ ] Quality looks good

### Performance Check
- [ ] Note inference time from test
- [ ] Is it acceptable? (250ms or less is good)
- [ ] Memory usage acceptable? (should be <200MB)

## Phase 5: Production Deployment

### Batch Processing (Optional)
- [ ] Prepare input image directory
- [ ] Transfer images to KRIA:
  ```bash
  scp -r input_images/ root@<kria-ip>:/root/
  ```
- [ ] Run batch processing:
  ```bash
  python3 /root/kria_predict.py --batch /root/input_images/ \
    --model /root/fastcghnet_lite.onnx \
    --output /root/output_batch/
  ```
- [ ] Check output directory
- [ ] All images processed successfully

### Benchmarking (Optional)
- [ ] Run benchmark to measure performance:
  ```bash
  python3 /root/kria_predict.py --benchmark \
    --model /root/fastcghnet_lite.onnx
  ```
- [ ] Note min/max/mean times
- [ ] Document performance results

### Error Handling
- [ ] Test error cases:
  - [ ] Non-existent model file
  - [ ] Missing input image
  - [ ] Invalid file formats
- [ ] Verify graceful error handling

## Phase 6: Optional - FPGA Acceleration

### Vitis AI Setup (Advanced)
- [ ] Decide if FPGA acceleration needed
- [ ] If YES, follow `KRIA_DEPLOYMENT.md` Vitis AI section
- [ ] Run vitis_ai_helper.py on dev machine:
  ```bash
  python vitis_ai_helper.py --install
  ```

### Model Quantization
- [ ] Prepare quantization environment
- [ ] Export quantized model
- [ ] Compile for FPGA
- [ ] Transfer compiled model to KRIA
- [ ] Test FPGA inference:
  ```bash
  python3 /root/kria_predict.py image.png --vitis-ai
  ```
- [ ] Verify significant speedup (40-50ms expected)

## Phase 7: Production Monitoring

### Ongoing Checks
- [ ] Monitor inference times
- [ ] Watch for memory leaks
- [ ] Check disk space usage
- [ ] Log performance metrics

### Backup & Maintenance
- [ ] Backup trained models
- [ ] Backup quantized models (if using FPGA)
- [ ] Keep deployment scripts versioned
- [ ] Document any modifications

## Phase 8: Troubleshooting

### If Something Goes Wrong

**ONNX Runtime Import Error:**
```bash
pip install onnxruntime-aarch64  # For 64-bit ARM
# or
pip install onnxruntime          # Auto-detect architecture
```

**Out of Memory:**
- [ ] Check available memory: `free -h`
- [ ] Use lite model (already using)
- [ ] Reduce batch size if processing batches

**Slow Inference:**
- [ ] Verify using lite model
- [ ] Check CPU load: `top` or `htop`
- [ ] Consider Vitis AI for FPGA speedup

**Connection Issues:**
- [ ] Verify KRIA is powered and connected
- [ ] Check network: `ping <kria-ip>`
- [ ] Verify SSH access: `ssh root@<kria-ip> echo ok`

**Model Not Found:**
- [ ] Check file exists: `ssh root@<kria-ip> ls -lh /root/*.onnx`
- [ ] Re-export if missing: `export_onnx(...)`
- [ ] Re-transfer file

## Final Verification Checklist

### Essential
- [ ] Model exports to ONNX without errors
- [ ] Single image inference works on KRIA
- [ ] Output files created correctly
- [ ] Performance acceptable (250ms or better)

### Recommended
- [ ] Batch processing tested
- [ ] Benchmarking completed
- [ ] Error cases handled gracefully
- [ ] Documentation reviewed

### Optional
- [ ] FPGA (Vitis AI) setup completed
- [ ] Performance optimization done
- [ ] Monitoring/logging configured
- [ ] Backup procedures established

## Quick Reference - Common Commands

```bash
# Export
python -c "from FastCGHNet import export_onnx; export_onnx()"

# Test locally
python kria_predict.py image.png --model model.onnx

# Transfer
scp model.onnx root@kria:/root/
scp kria_predict.py root@kria:/root/

# Setup
ssh root@kria bash setup_kria.sh

# Test on KRIA
ssh root@kria python /root/kria_predict.py /root/image.png

# Benchmark
ssh root@kria python /root/kria_predict.py --benchmark

# Batch
ssh root@kria python /root/kria_predict.py --batch /root/inputs/ \
  --model /root/model.onnx --output /root/outputs/
```

## Success Indicators

✅ Model exported successfully
✅ Files transferred to KRIA
✅ KRIA setup completed without errors
✅ Single image inference works
✅ Output files created and valid
✅ Inference time acceptable
✅ Memory usage acceptable
✅ Ready for production use!

---

**Estimated Total Time:**
- Fast track (CPU only): 1-2 hours
- Full setup (with FPGA): 4-6 hours

**Need Help?**
- See `KRIA_README.md` for complete guide
- See `KRIA_DEPLOYMENT.md` for technical details
- See `TROUBLESHOOTING` section above
