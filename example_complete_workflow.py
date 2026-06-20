#!/usr/bin/env python3
"""
Complete example: Process images and train FastCGHNet
This script demonstrates the full workflow from raw images to trained model.

Usage:
    python example_complete_workflow.py ~/my_images
    python example_complete_workflow.py ~/my_images --epochs 20 --batch-size 4
"""

import sys
from pathlib import Path
import argparse


def example_basic_processing():
    """Example 1: Basic image processing."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Image Processing")
    print("="*70)
    
    print("""
# Process images to 1-bit quantized format
from image_to_training_data import process_input_folder

results = process_input_folder(
    input_dir="~/my_images",
    output_dir="./training_data",
    target_width=1345,
    target_height=800,
    verbose=True
)

print(f"Processed {len(results)} images")
for r in results:
    print(f"  {r['input']}: {r['packed_size']/1024:.1f} KB")
""")


def example_dataset_loading():
    """Example 2: Loading processed data with PyTorch."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Dataset Loading")
    print("="*70)
    
    print("""
# Load processed images as PyTorch dataset
from FastCGHNet import QuantizedImageDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = QuantizedImageDataset(
    data_dir="./training_data",
    max_samples=500,
    device="cuda",
    generate_phase=True
)

print(f"Dataset size: {len(dataset)}")

# Create dataloader for batching
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# Iterate through batches
for batch_idx, (images, phases) in enumerate(loader):
    print(f"Batch {batch_idx}: images={images.shape}, phases={phases.shape}")
    if batch_idx >= 2:  # Show first 3 batches
        break
""")


def example_training():
    """Example 3: Training the network."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Training FastCGHNet")
    print("="*70)
    
    print("""
# Option A: Simple training with new dataset
from FastCGHNet import train_cgh_network

model = train_cgh_network(
    use_quantized_data=True,
    quantized_data_dir="./training_data",
    output_dir="./models",
    num_epochs=20,
    batch_size=4,
    learning_rate=5e-4,
    max_samples=None  # Use all images
)

# Option B: Training with old dataset (backward compatible)
model = train_cgh_network(
    samples_dir="/Users/Ish/Hologram/Training Data/samples",
    output_dir="./models",
    num_epochs=20,
    batch_size=2
)

# Option C: Custom training loop
import torch
from FastCGHNet import FastCGHNet, QuantizedImageDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastCGHNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()

dataset = QuantizedImageDataset("./training_data", device=device)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(20):
    for images, phases in loader:
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, phases)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
""")


def example_inference():
    """Example 4: Using the trained model."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Inference / Prediction")
    print("="*70)
    
    print("""
# Use trained model to generate holograms
from FastCGHNet import predict_hologram
from PIL import Image

# Method 1: Predict from image file
phase, cgh = predict_hologram(
    "~/my_images/photo.jpg",
    model_path="./models/best_model.pt",
    output_path="./output/cgh_photo.bmp"
)

# Method 2: Batch processing
from batch_process import batch_process

batch_process(
    input_dir="~/my_images",
    output_dir="./output_holograms",
    model_path="./models/best_model.pt"
)

# Method 3: Direct model prediction
import torch
from FastCGHNet import FastCGHNet
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastCGHNet().to(device)

# Load image
img = Image.open("photo.jpg").convert('L')
img_array = np.array(img, dtype=np.float32) / 255.0
img_tensor = torch.from_numpy(img_array[np.newaxis, np.newaxis]).to(device)

# Predict
with torch.no_grad():
    phase = model(img_tensor)

print(f"Generated phase: shape={phase.shape}, range=[{phase.min():.2f}, {phase.max():.2f}]")
""")


def example_full_workflow():
    """Example 5: Complete end-to-end workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete End-to-End Workflow")
    print("="*70)
    
    print("""
# Complete workflow from images to holograms

# Step 1: Process images
print("Step 1: Processing images...")
from image_to_training_data import process_input_folder

results = process_input_folder(
    input_dir="~/my_images",
    output_dir="./training_data",
    verbose=True
)
print(f"✓ Processed {len(results)} images")

# Step 2: Train network
print("\\nStep 2: Training network...")
from FastCGHNet import train_cgh_network

train_cgh_network(
    use_quantized_data=True,
    quantized_data_dir="./training_data",
    num_epochs=10,
    batch_size=2,
    output_dir="./models"
)
print("✓ Training complete")

# Step 3: Generate holograms
print("\\nStep 3: Generating holograms...")
from FastCGHNet import predict_hologram

for image_path in Path("~/my_images").glob("*.jpg"):
    output = Path("./output") / (image_path.stem + "_cgh.bmp")
    predict_hologram(
        str(image_path),
        model_path="./models/best_model.pt",
        output_path=str(output)
    )
    print(f"✓ Generated: {output}")

print("\\n✓ Complete workflow finished!")
print(f"Holograms saved to: ./output/")
""")


def example_comparison():
    """Example 6: Old vs New approach."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Old vs New Data Format")
    print("="*70)
    
    print("""
OLD APPROACH (Still supported):
    Pre-processed training data with known ground truth phase
    
    train_cgh_network(
        samples_dir="/Users/Ish/Hologram/Training Data/samples",
        num_epochs=30
    )
    
    ✓ Advantages: Known ground truth, proven performance
    ✗ Disadvantages: Limited data, requires pre-processing

NEW APPROACH:
    Raw images → Process → Train with synthetic phase
    
    process_input_folder("~/my_images", "./my_data")
    train_cgh_network(
        use_quantized_data=True,
        quantized_data_dir="./my_data",
        num_epochs=30
    )
    
    ✓ Advantages: Custom data, scalable, flexible
    ✗ Disadvantages: Synthetic phase, may need tuning
    
COMBINING BOTH:
    Use old data for initial training, then fine-tune on new data
    
    # Train on known data
    train_cgh_network(samples_dir="./Training Data/samples", num_epochs=10)
    
    # Fine-tune on new data
    train_cgh_network(
        use_quantized_data=True,
        quantized_data_dir="./my_data",
        num_epochs=5
    )
""")


def example_performance():
    """Example 7: Performance monitoring."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Performance Monitoring")
    print("="*70)
    
    print("""
# Monitor processing and training performance

import time
from image_to_training_data import process_input_folder

# Process with timing
start = time.time()
results = process_input_folder("~/my_images", "./data")
elapsed = time.time() - start

print(f"Processing Performance:")
print(f"  Images: {len(results)}")
print(f"  Time: {elapsed:.1f}s")
print(f"  Speed: {len(results)/elapsed:.1f} images/sec")
print(f"  Total size: {sum(r['packed_size'] for r in results) / 1024 / 1024:.1f} MB")
print(f"  Avg size: {sum(r['packed_size'] for r in results) / len(results) / 1024:.1f} KB")

# Training with monitoring
import torch
from FastCGHNet import FastCGHNet, QuantizedImageDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastCGHNet().to(device)
dataset = QuantizedImageDataset("./data", device=device)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

start = time.time()
total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTraining Setup:")
print(f"  Device: {device}")
print(f"  Model parameters: {total_params:,}")
print(f"  Dataset size: {len(dataset)}")
print(f"  Batches: {len(loader)}")
print(f"  Batch size: 4")

# Sample batch processing
batch_start = time.time()
for images, phases in loader:
    with torch.no_grad():
        pred = model(images)
    break
batch_time = time.time() - batch_start

print(f"  Batch time: {batch_time*1000:.1f}ms")
print(f"  Throughput: {len(images)/batch_time:.1f} images/sec")
""")


def main():
    """Print all examples."""
    examples = [
        example_basic_processing,
        example_dataset_loading,
        example_training,
        example_inference,
        example_full_workflow,
        example_comparison,
        example_performance,
    ]
    
    print("\n" + "="*70)
    print("FASTCGHNET - COMPLETE WORKFLOW EXAMPLES")
    print("="*70)
    print("""
This file shows 7 complete examples of using the new image processing
pipeline with FastCGHNet.

Each example is self-contained and can be copy-pasted into your code.
""")
    
    for example_func in examples:
        example_func()
    
    print("\n" + "="*70)
    print("QUICK START")
    print("="*70)
    print("""
Ready to get started? Run these commands:

1. Process your images:
   python image_to_training_data.py ~/my_images -o ./training_data -v

2. Train the model:
   python quickstart.py ~/my_images --epochs 10

3. Check the results:
   python validate_pipeline.py

For more information:
   - See QUICKSTART_GUIDE.md for simple usage
   - See IMAGE_PROCESSING_README.md for technical details
   - Run: python example_complete_workflow.py

Happy training! 🎉
""")


if __name__ == "__main__":
    main()
