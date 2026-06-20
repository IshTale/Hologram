#!/usr/bin/env python3
"""
Generate a hologram from a training data image using PLM.py
"""

from PLM import CGHGenerator, DeviceLibrary
import numpy as np
from PIL import Image

# Setup
training_sample = "/Users/Ish/Hologram/Training Data/samples/sample_000000"
input_image = f"{training_sample}/view_1.bmp"
output_dir = "/Users/Ish/Hologram/output"

# Create device and generator
device_library = DeviceLibrary()
device = device_library.defineDevice("0.67")

generator = CGHGenerator()

print(f"Loading image from: {input_image}")
print(f"Device: {device['device']}")
print(f"Resolution: {device['w']} x {device['h']}")
print(f"Phase levels: {device['nLevel']}")
print()

# Generate hologram
print("Running ADAMWGS optimization...")
generator.createCGH(
    device,
    filename=input_image,
    colorChannel=0,
    alg="ADAMWGS",
    numIter=100,
    initialPhase="Random",
    propMethod="Fourier",
    ShiftFOV=True,
    showImages=False,
    binarizeTarget=False,
    preserveAspect=True,
    lossMode="auto"
)

print()
print("Optimization complete!")
print(f"CGH phase shape: {generator.CGH_output_phase_disc.shape}")
print(f"CGH mapped shape: {generator.CGH_mapped.shape}")

# Save outputs
import os
os.makedirs(output_dir, exist_ok=True)

cgh_output = f"{output_dir}/cgh_from_training.bmp"
generator.writeCGHToFile(cgh_output, binary=False)
print(f"\nWrote CGH to: {cgh_output}")

# Also save the recovered image for comparison
recovered_output = f"{output_dir}/recovered_image.png"
recovered_img = Image.fromarray((generator.imRecovered_disc * 255).astype(np.uint8))
recovered_img.save(recovered_output)
print(f"Wrote recovered image to: {recovered_output}")

print("\nDone!")
