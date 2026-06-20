#!/usr/bin/env python3
"""
Quick inference script for FastCGHNet
Use this to predict holograms in milliseconds instead of seconds of optimization
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import time

# Add repo to path
sys.path.insert(0, '/Users/Ish/Hologram')
from FastCGHNet import FastCGHNet
from PLM import DeviceLibrary, CGHGenerator


def fast_predict(image_path, model_path=None, output_bmp=None, output_phase=None):
    """
    Predict hologram instantly using neural network
    
    Args:
        image_path: Input image file
        model_path: Path to trained model (downloads/trains if needed)
        output_bmp: Save CGH as BMP
        output_phase: Save phase as NPY
    
    Returns:
        phase_np: Predicted phase (800, 1358)
        cgh_mapped: Device-formatted hologram
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if model_path is None:
        model_path = "/Users/Ish/Hologram/models/best_model.pt"
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"⚠️  Model not found at {model_path}")
        print("Train with: python FastCGHNet.py train")
        return None, None
    
    model = FastCGHNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    
    # Load and preprocess image
    img_pil = Image.open(image_path).convert('L')
    img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    
    # Resize to device size
    if img_array.shape != (800, 1358):
        print(f"  Resizing from {img_array.shape} to (800, 1358)")
        img_array = cv2.resize(img_array, (1358, 800), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    
    img_tensor = torch.from_numpy(img_array[np.newaxis, np.newaxis, :, :]).to(device)
    
    # Predict
    print(f"Predicting hologram phase...")
    t0 = time.time()
    with torch.no_grad():
        phase_pred = model(img_tensor)
    t_pred = time.time() - t0
    
    phase_np = phase_pred.squeeze().cpu().numpy()
    
    print(f"✓ Prediction time: {t_pred*1000:.1f}ms (vs ~20s for iterative)")
    print(f"  Phase range: [{phase_np.min():.3f}, {phase_np.max():.3f}]")
    
    # Format for device
    device_lib = DeviceLibrary()
    device_dict = device_lib.defineDevice("0.67")
    
    phase_disc, state_disc = CGHGenerator.discretePhase(
        phase_np, device_dict["nLevel"], device_dict["pLevel"]
    )
    cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
    
    # Save outputs
    if output_bmp:
        cgh_uint8 = cv2.normalize(cgh_mapped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(output_bmp, cgh_uint8)
        print(f"✓ Wrote CGH BMP: {output_bmp}")
    
    if output_phase:
        np.save(output_phase, phase_np)
        print(f"✓ Wrote phase: {output_phase}")
    
    return phase_np, cgh_mapped


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast hologram prediction with neural network")
    parser.add_argument("image", help="Input image file")
    parser.add_argument("--model", default="/Users/Ish/Hologram/models/best_model.pt", help="Model path")
    parser.add_argument("--output-bmp", help="Save CGH as BMP")
    parser.add_argument("--output-phase", help="Save phase as NPY")
    
    args = parser.parse_args()
    
    fast_predict(
        args.image,
        model_path=args.model,
        output_bmp=args.output_bmp,
        output_phase=args.output_phase,
    )
