# Image Processing Pipeline for Hologram Training

This document describes the new image processing pipeline that converts raw input images to 1-bit quantized training data.

## Overview

The pipeline processes regular images (JPEG, PNG, BMP, etc.) and converts them to:
1. **1-bit quantized bitmap** - Binary images (0 or 1 only)
2. **1345 x 800 resolution** - Resized with aspect ratio preservation
3. **Efficient packed format** - NPZ files with bit-packed storage

This preprocessing step prepares images for training the FastCGHNet neural network.

## Pipeline Components

### 1. Image Processing Script (`image_to_training_data.py`)

Converts raw images to 1-bit quantized training data.

**Usage:**
```bash
python image_to_training_data.py <input_directory> [options]
```

**Arguments:**
- `input_directory`: Directory containing input images
- `-o, --output`: Output directory (default: `./training_data_1bit`)
- `-w, --width`: Target width (default: 1345)
- `-h, --height`: Target height (default: 800)
- `-v, --verify`: Verify output after processing

**Example:**
```bash
python image_to_training_data.py ~/my_images -o ./training_data -v
```

**Processing Steps:**
1. Loads each image and converts to grayscale
2. Resizes to target dimensions (1345 x 800) with aspect ratio preservation
   - Adds padding to maintain aspect ratio
   - Centers the image in the output canvas
3. Quantizes to 1-bit (binary):
   - Threshold at 0.5 (pixel values > 0.5 → 1, else → 0)
4. Packs bits into bytes for efficient storage
5. Saves as compressed NPZ file

**Output Format:**
Each processed image creates an NPZ file containing:
- `packed_bits`: Bit-packed byte array (height, ceil(width/8))
- `shape`: Original dimensions (height, width)
- `original_path`: Source file path for reference

### 2. Dataset Classes (`FastCGHNet.py`)

Two dataset classes for training:

#### `HologramDataset` (Existing)
- Loads pre-processed training samples with ground truth phase
- Format: Directory with `input_views_packed.npz` and `cgh_phase_cont_float32.npy`
- Used for standard training with known hologram data

#### `QuantizedImageDataset` (New)
- Loads 1-bit quantized images from NPZ files
- Generates synthetic phase ground truth for training
- Flexible phase generation strategy

**Usage in training:**
```python
from FastCGHNet import QuantizedImageDataset, train_cgh_network

# Option 1: Use new quantized image dataset
dataset = QuantizedImageDataset(
    data_dir="./training_data",
    max_samples=1000,
    device="cuda"
)

# Option 2: Train directly with new data source
train_cgh_network(
    use_quantized_data=True,
    quantized_data_dir="./training_data",
    num_epochs=30,
    batch_size=2
)
```

## Workflow Example

### Step 1: Prepare Input Images
```bash
mkdir ~/my_hologram_images
# Copy your images (JPG, PNG, BMP, etc.) to ~/my_hologram_images
```

### Step 2: Process Images
```bash
cd /Users/Ish/Hologram
python image_to_training_data.py ~/my_hologram_images -o ./training_data_1bit -v
```

**Output:**
```
======================================================================
Image Processing and Quantization
======================================================================
Input directory:    ~/my_hologram_images
Output directory:   ./training_data_1bit
Target resolution:  1345 x 800
Output format:      1-bit bitmap (packed bytes)
Images to process:  150
======================================================================

Processing: 100%|████████████| 150/150 [00:45<00:00, 3.33it/s]

======================================================================
PROCESSING COMPLETE
======================================================================
Processed:  150/150
Total size: 12.5 MB
Avg/file:   85.3 KB
======================================================================
```

### Step 3: Verify Processed Data
```bash
python image_to_training_data.py ~/my_hologram_images -o ./training_data_1bit -v
```

The verification shows sample statistics:
```
Verifying 3 sample(s)...

File: image_000000.npz
  Packed shape:   (800, 169)
  Original shape: (800, 1345)
  Image dims:     800 x 1345
  Unique values:  [0 1]
  File size:      47.3 KB
```

### Step 4: Train Network
```python
from FastCGHNet import QuantizedImageDataset, train_cgh_network

# Train with new quantized images
train_cgh_network(
    use_quantized_data=True,
    quantized_data_dir="./training_data_1bit",
    num_epochs=30,
    batch_size=2
)
```

## Image Quantization Details

### What is 1-bit Quantization?
- Each pixel is either 0 (white/off) or 1 (black/on)
- No gradients or colors - binary only
- Similar to scanning a black and white document

### Threshold Operation
- Input: Grayscale image (0-255)
- Apply threshold at 0.5 (normalized: 0.5 on 0-1 scale)
- Output: Binary image (0 or 1)

**Example:**
```
Input:  [0, 64, 128, 192, 255]  # Grayscale
        [0, 0.25, 0.5, 0.75, 1.0]  # Normalized
Threshold at 0.5:
Output: [0, 0, 0, 1, 1]  # Binary
```

### Resizing with Aspect Ratio
- Target: 1345 x 800
- Method: Thumbnail + center padding
- Preserves original image aspect ratio
- Fills remaining space with white (255)

**Example:**
```
Original: 2000 x 1000 (2:1 ratio)
↓ (resize to fit)
Resized:  1600 x 800 (2:1 ratio maintained)
↓ (center in canvas)
Final:    1345 x 800 (padded left/right)
```

### Bit Packing
- Stores 8 binary pixels per byte
- Reduces storage from 1.34 MB to ~170 KB per image
- Uses big-endian bit order for consistency

**Example:**
```
8 pixels: [1, 0, 1, 1, 0, 0, 1, 0]
Packed:   0b10110010 = 178
```

## File Organization

```
/Users/Ish/Hologram/
├── image_to_training_data.py          # Image processing script
├── FastCGHNet.py                      # Updated with QuantizedImageDataset
├── training_data_1bit/                # Output directory (created by script)
│   ├── image_000000.npz
│   ├── image_000001.npz
│   ├── image_000002.npz
│   └── ...
├── Training Data/                     # Old format (still supported)
│   └── samples/
│       ├── sample_000000/
│       │   ├── input_views_packed.npz
│       │   └── cgh_phase_cont_float32.npy
│       └── ...
└── models/
    └── best_model.pt                  # Trained model
```

## Performance Notes

### Processing Speed
- Typical: 3-5 images/second on modern CPU
- Depends on image resolution and format
- GPU acceleration not needed for preprocessing

### Storage Efficiency
- Original image: 2-10 MB (typical JPEG)
- Processed (1-bit): 47-85 KB (packed)
- Compression ratio: ~99%

### Memory Requirements
- Per-image RAM: ~10 MB (during processing)
- Full dataset RAM: ~50-100 MB (for typical 500-1000 images)

## Advanced Usage

### Custom Resolution
```bash
python image_to_training_data.py ~/images -w 1280 -h 720
```

### Batch Processing
```bash
# Process multiple folders
for folder in ~/images/*; do
    python image_to_training_data.py "$folder" -o "./training_data/$(basename $folder)"
done
```

### Integration with Training
```python
from FastCGHNet import QuantizedImageDataset, DataLoader, train_cgh_network
import torch

# Load dataset
dataset = QuantizedImageDataset("./training_data_1bit", max_samples=500)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Use in training loop
for epoch in range(30):
    for img_batch, phase_batch in dataloader:
        # Training code here
        pass
```

## Troubleshooting

### No images found
```
❌ No image files found in ~/my_images
   Searched for patterns: *.jpg, *.jpeg, *.png, *.bmp, *.tiff, *.gif
```
**Solution:** Verify input directory exists and contains supported image formats

### Memory issues during processing
```
MemoryError: Unable to allocate X GB for processing
```
**Solution:** Reduce batch size or process images in smaller folders

### Verification shows wrong dimensions
```
Error: Expected shape (800, 1345) but got (800, 1200)
```
**Solution:** Check input images - some may be smaller than target resolution

## See Also

- `PLM.py` - Holographic phase computation
- `batch_process.py` - Batch inference script
- `FastCGHNet.py` - Neural network model and training

## References

- Bit packing: Used for efficient storage of binary data
- Aspect ratio preservation: Common in image processing pipelines
- 1-bit quantization: Standard preprocessing for binary neural networks
