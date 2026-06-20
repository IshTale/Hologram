#!/bin/bash
# FastCGHNet - Quick Command Reference
# Copy and paste any command to run

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING (Run once to train the model)
# ═══════════════════════════════════════════════════════════════════════════════

# Start training (already running in background)
cd /Users/Ish/Hologram && python FastCGHNet.py train

# Monitor training progress
tail -f /Users/Ish/Hologram/models/training.log

# Check if model file exists
ls -lh /Users/Ish/Hologram/models/best_model.pt


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE (Fast hologram generation)
# ═══════════════════════════════════════════════════════════════════════════════

# Process single image
cd /Users/Ish/Hologram && python fast_predict.py \
  /Users/Ish/Hologram/Training\ Data/samples/sample_000000/view_1.bmp \
  --output-bmp /Users/Ish/Hologram/output/sample_0_cgh.bmp

# Process any image
cd /Users/Ish/Hologram && python fast_predict.py /path/to/image.bmp \
  --output-bmp /path/to/output.bmp

# Process and save phase as NPY
cd /Users/Ish/Hologram && python fast_predict.py image.bmp \
  --output-bmp output.bmp \
  --output-phase output_phase.npy


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

# Process entire directory
cd /Users/Ish/Hologram && python batch_process.py \
  --input /Users/Ish/Hologram/Training\ Data/samples/sample_000000 \
  --output /Users/Ish/Hologram/output/batch_results

# Process all training samples
cd /Users/Ish/Hologram && python batch_process.py --training-samples

# Process first 10 training samples
cd /Users/Ish/Hologram && python batch_process.py --training-samples --max 10

# Process first 100 training samples
cd /Users/Ish/Hologram && python batch_process.py --training-samples --max 100


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKING (Compare speed: PLM vs FastCGHNet)
# ═══════════════════════════════════════════════════════════════════════════════

# Quick benchmark (20 PLM iterations)
cd /Users/Ish/Hologram && python benchmark.py \
  /Users/Ish/Hologram/Training\ Data/samples/sample_000000/view_1.bmp

# Benchmark with 100 PLM iterations
cd /Users/Ish/Hologram && python benchmark.py \
  /Users/Ish/Hologram/Training\ Data/samples/sample_000000/view_1.bmp \
  --iter 100

# Benchmark with custom model path
cd /Users/Ish/Hologram && python benchmark.py image.bmp \
  --model /path/to/custom/model.pt


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON API (Use in your own scripts)
# ═══════════════════════════════════════════════════════════════════════════════

cat > test_api.py << 'PYEOF'
import sys
sys.path.insert(0, '/Users/Ish/Hologram')

from fast_predict import fast_predict
from pathlib import Path

# Single image
phase, cgh = fast_predict(
    "/Users/Ish/Hologram/Training Data/samples/sample_000000/view_1.bmp",
    output_bmp="/Users/Ish/Hologram/output/api_test.bmp"
)
print(f"Phase shape: {phase.shape}")
print(f"CGH shape: {cgh.shape}")

# Batch process
for sample_dir in Path("/Users/Ish/Hologram/Training Data/samples").iterdir()[:10]:
    img = sample_dir / "view_1.bmp"
    if img.exists():
        output = f"/Users/Ish/Hologram/output/{sample_dir.name}_cgh.bmp"
        fast_predict(str(img), output_bmp=output)
        print(f"✓ {sample_dir.name}")
PYEOF

python test_api.py


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUGGING & TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════

# Check all files
ls -lh /Users/Ish/Hologram/{FastCGHNet,fast_predict,batch_process,benchmark}.py

# Check documentation
ls -lh /Users/Ish/Hologram/{QUICKSTART,FASTCGHNET_README,IMPLEMENTATION_SUMMARY}.*

# Test imports
python -c "from FastCGHNet import FastCGHNet; print('✓ FastCGHNet imported')"

# Check model parameters
python -c "from FastCGHNet import FastCGHNet; m = FastCGHNet(); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')"

# List trained models
ls -lh /Users/Ish/Hologram/models/

# Show training loss trend
grep "Epoch.*Avg Loss" /Users/Ish/Hologram/models/training.log | tail -10


# ═══════════════════════════════════════════════════════════════════════════════
# STATS & MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

# Training progress (last 20 lines)
tail -20 /Users/Ish/Hologram/models/training.log

# Count completed epochs
grep "Epoch.*Avg Loss" /Users/Ish/Hologram/models/training.log | wc -l

# Best loss so far
grep "Epoch.*Avg Loss" /Users/Ish/Hologram/models/training.log | tail -1

# File sizes
du -sh /Users/Ish/Hologram/{FastCGHNet,fast_predict,batch_process,benchmark}.py

# Output statistics
echo "Generated holograms:"
ls -1 /Users/Ish/Hologram/output/*.bmp 2>/dev/null | wc -l


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED USAGE
# ═══════════════════════════════════════════════════════════════════════════════

# Process with custom model
python fast_predict.py image.bmp \
  --model /custom/path/model.pt \
  --output-bmp output.bmp

# Batch with custom model
python batch_process.py \
  --input ./images \
  --output ./outputs \
  --model /custom/path/model.pt

# Process with different pattern
python batch_process.py \
  --input ./images \
  --output ./outputs \
  --pattern "*.png"  # or "*.jpg" etc


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

# View quick start
cat /Users/Ish/Hologram/QUICKSTART.md

# View detailed docs
cat /Users/Ish/Hologram/FASTCGHNET_README.md

# View implementation summary
cat /Users/Ish/Hologram/IMPLEMENTATION_SUMMARY.txt

# View this file
cat /Users/Ish/Hologram/COMMANDS.sh
