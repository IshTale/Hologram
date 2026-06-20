#!/usr/bin/env python3
"""
Compare inference speed: PlM.py (iterative) vs FastCGHNet (neural network)
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, '/Users/Ish/Hologram')

from PLM import CGHGenerator, DeviceLibrary
from fast_predict import fast_predict
import numpy as np


def benchmark_plm(image_path, num_iter=20):
    """Benchmark PLM.py optimization speed"""
    
    device_library = DeviceLibrary()
    device = device_library.defineDevice("0.67")
    generator = CGHGenerator()
    
    print(f"\n{'='*60}")
    print(f"PLM.py (ADAMWGS) - {num_iter} iterations")
    print(f"{'='*60}")
    
    t0 = time.time()
    generator.createCGH(
        device,
        filename=image_path,
        colorChannel=0,
        alg="ADAMWGS",
        numIter=num_iter,
        initialPhase="Random",
        propMethod="Fourier",
        ShiftFOV=True,
        showImages=False,
        binarizeTarget=False,
        preserveAspect=True,
        lossMode="auto"
    )
    t_total = time.time() - t0
    
    print(f"\nTotal time: {t_total:.2f}s")
    print(f"Time per iteration: {t_total/num_iter*1000:.1f}ms")
    print(f"Phase shape: {generator.CGH_output_phase_disc.shape}")
    
    return generator.CGH_output_phase_disc, t_total


def benchmark_fastcghnet(image_path, model_path="/Users/Ish/Hologram/models/best_model.pt"):
    """Benchmark FastCGHNet inference speed"""
    
    print(f"\n{'='*60}")
    print(f"FastCGHNet (Neural Network)")
    print(f"{'='*60}")
    
    t0 = time.time()
    phase, _ = fast_predict(
        image_path,
        model_path=model_path,
        output_bmp=None,
        output_phase=None
    )
    t_total = time.time() - t0
    
    print(f"\nTotal time: {t_total*1000:.1f}ms")
    print(f"Phase shape: {phase.shape}")
    
    return phase, t_total


def compare(image_path, model_path=None, num_iter_plm=20):
    """Compare both methods"""
    
    print("\n" + "="*60)
    print("HOLOGRAM GENERATION SPEED COMPARISON")
    print("="*60)
    print(f"Input: {image_path}")
    
    if model_path is None:
        model_path = "/Users/Ish/Hologram/models/best_model.pt"
    
    # Run FastCGHNet
    try:
        phase_nn, t_nn = benchmark_fastcghnet(image_path, model_path)
    except Exception as e:
        print(f"\n❌ FastCGHNet error: {e}")
        phase_nn = None
        t_nn = float('inf')
    
    # Run PLM
    try:
        phase_plm, t_plm = benchmark_plm(image_path, num_iter=num_iter_plm)
    except Exception as e:
        print(f"\n❌ PLM error: {e}")
        phase_plm = None
        t_plm = float('inf')
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"FastCGHNet:        {t_nn*1000:8.1f}ms")
    print(f"PLM.py ({num_iter_plm} iter):  {t_plm:8.1f}s")
    print(f"\n✨ Speedup: {t_plm/t_nn:,.0f}x faster with FastCGHNet")
    
    # Quality comparison (if both ran)
    if phase_nn is not None and phase_plm is not None:
        mse = np.mean((phase_nn - phase_plm) ** 2)
        print(f"Phase difference MSE: {mse:.6f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark hologram generation")
    parser.add_argument("image", help="Input image file")
    parser.add_argument("--model", default="/Users/Ish/Hologram/models/best_model.pt", help="Model path")
    parser.add_argument("--iter", type=int, default=20, help="PLM iterations (lower = faster but lower quality)")
    
    args = parser.parse_args()
    
    compare(args.image, model_path=args.model, num_iter_plm=args.iter)
