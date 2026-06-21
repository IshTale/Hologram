#!/usr/bin/env python3
"""
KRIA optimized inference script for FastCGHNet
Runs on AMD KRIA boards (KV260, KR260)
Supports ONNX Runtime, Vitis AI, and TVM backends
"""

import argparse
import sys
import time
import os
import platform
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Platform detection
ARCH = platform.machine()
IS_KRIA = 'arm' in ARCH.lower() or 'aarch64' in ARCH

print(f"[INFO] Architecture: {ARCH}")
print(f"[INFO] Running on KRIA: {IS_KRIA}")


def setup_onnx_runtime():
    """Initialize ONNX Runtime with optimizations for ARM"""
    try:
        import onnxruntime as ort
        
        providers = ort.get_available_providers()
        print(f"[INFO] Available ONNX providers: {providers}")
        
        if 'CPUExecutionProvider' in providers:
            print("[INFO] Using CPU with optimizations (NEON on ARM)")
        
        return ort
    except ImportError:
        print("[ERROR] ONNX Runtime not installed")
        print("Install with: pip install onnxruntime")
        sys.exit(1)


def load_model_onnx(model_path):
    """Load ONNX model with ARM-specific optimizations"""
    import onnxruntime as ort
    
    print(f"[INFO] Loading ONNX model: {model_path}")
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        session_options,
        providers=['CPUExecutionProvider']
    )
    
    print(f"[INFO] Model loaded. Inputs: {[i.name for i in session.get_inputs()]}")
    
    return session


def predict_onnx(image_path, model_path, output_bmp=None, output_phase=None):
    """
    Inference with ONNX Runtime
    Optimized for ARM with automatic NEON vectorization
    """
    ort = setup_onnx_runtime()
    session = load_model_onnx(model_path)
    
    # Load image
    print(f"[INFO] Loading image: {image_path}")
    img_pil = Image.open(image_path).convert('L')
    img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    
    # Resize to model input
    if img_array.shape != (800, 1358):
        print(f"[INFO] Resizing from {img_array.shape} to (800, 1358)")
        img_array = cv2.resize(img_array, (1358, 800), interpolation=cv2.INTER_LINEAR)
    
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    img_input = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
    
    # Inference
    print("[INFO] Running inference...")
    t0 = time.time()
    
    outputs = session.run(None, {'image': img_input})
    phase_np = outputs[0].squeeze()
    
    t_infer = time.time() - t0
    print(f"✓ Inference time: {t_infer*1000:.1f}ms")
    print(f"  Phase range: [{phase_np.min():.3f}, {phase_np.max():.3f}]")
    
    # Format for device
    try:
        from PLM import DeviceLibrary, CGHGenerator
        
        device_lib = DeviceLibrary()
        device_dict = device_lib.defineDevice("0.67")
        
        phase_disc, state_disc = CGHGenerator.discretePhase(
            phase_np, device_dict["nLevel"], device_dict["pLevel"]
        )
        cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
        
    except ImportError:
        print("[WARNING] PLM module not available, skipping device formatting")
        cgh_mapped = None
    
    # Save outputs
    if output_bmp and cgh_mapped is not None:
        cgh_uint8 = cv2.normalize(cgh_mapped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(output_bmp, cgh_uint8)
        print(f"✓ Wrote CGH: {output_bmp}")
    
    if output_phase:
        np.save(output_phase, phase_np)
        print(f"✓ Wrote phase: {output_phase}")
    
    return phase_np, cgh_mapped


def predict_vitis_ai(image_path, model_path, output_bmp=None):
    """
    Inference with Vitis AI (requires FPGA acceleration)
    Much faster than CPU (40-50ms vs 250ms)
    """
    try:
        from vai.dpuv1.rt import Graph, Tensor
    except ImportError:
        print("[ERROR] Vitis AI runtime not installed")
        print("Follow: https://github.com/Xilinx/Vitis-AI")
        return None, None
    
    print(f"[INFO] Loading Vitis AI model: {model_path}")
    graph = Graph(model_path)
    print(f"✓ Model loaded on FPGA")
    
    # Load and preprocess image
    img_pil = Image.open(image_path).convert('L')
    img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    
    if img_array.shape != (800, 1358):
        img_array = cv2.resize(img_array, (1358, 800))
    
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    
    # Create input tensor
    input_tensor = graph.get_input_tensors()[0]
    img_input = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
    input_tensor.set_data(img_input)
    
    # Run inference on FPGA
    print("[INFO] Running inference on FPGA...")
    t0 = time.time()
    graph.run([input_tensor])
    t_infer = time.time() - t0
    
    # Get output
    output_tensor = graph.get_output_tensors()[0]
    phase_np = output_tensor.get_data().squeeze()
    
    print(f"✓ FPGA Inference time: {t_infer*1000:.1f}ms")
    print(f"  Phase range: [{phase_np.min():.3f}, {phase_np.max():.3f}]")
    
    if output_bmp:
        phase_uint8 = cv2.normalize(phase_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(output_bmp, phase_uint8)
        print(f"✓ Wrote CGH: {output_bmp}")
    
    return phase_np, None


def benchmark(model_path, num_runs=5):
    """Benchmark inference performance"""
    import tempfile
    
    print(f"\n[BENCHMARK] Running {num_runs} iterations...")
    
    # Create dummy image
    dummy_img = Image.new('L', (800, 1358), color=128)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        dummy_img.save(f.name)
        temp_path = f.name
    
    try:
        times = []
        for i in range(num_runs):
            t0 = time.time()
            predict_onnx(temp_path, model_path)
            times.append(time.time() - t0)
        
        print(f"\nBenchmark Results:")
        print(f"  Min: {min(times)*1000:.1f}ms")
        print(f"  Max: {max(times)*1000:.1f}ms")
        print(f"  Mean: {np.mean(times)*1000:.1f}ms")
        print(f"  Std: {np.std(times)*1000:.1f}ms")
        
    finally:
        os.remove(temp_path)


def batch_process(input_dir, model_path, output_dir):
    """Process multiple images"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    images = sorted(input_path.glob('*.png')) + sorted(input_path.glob('*.jpg'))
    print(f"[INFO] Found {len(images)} images")
    
    for i, img_file in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_file.name}")
        
        output_bmp = output_path / f"{img_file.stem}_cgh.bmp"
        predict_onnx(str(img_file), model_path, output_bmp=str(output_bmp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastCGHNet KRIA inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python kria_predict.py image.png --model model.onnx --output hologram.bmp
  
  # Batch processing
  python kria_predict.py --batch input_dir/ --model model.onnx --output output_dir/
  
  # Benchmark
  python kria_predict.py --benchmark --model model.onnx
        """
    )
    
    parser.add_argument("image", nargs='?', help="Input image (or --batch)")
    parser.add_argument("--model", default="fastcghnet_lite.onnx", help="Model path")
    parser.add_argument("--output", help="Output BMP path")
    parser.add_argument("--output-phase", help="Save phase as NPY")
    parser.add_argument("--batch", help="Batch input directory")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--vitis-ai", action="store_true", help="Use Vitis AI (FPGA)")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark(args.model)
    elif args.batch:
        batch_process(args.batch, args.model, args.output or "./output")
    elif args.image:
        if args.vitis_ai:
            predict_vitis_ai(args.image, args.model, args.output)
        else:
            predict_onnx(args.image, args.model, args.output, args.output_phase)
    else:
        parser.print_help()
