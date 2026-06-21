#!/usr/bin/env python3
"""
Fast inference for FastCGHNet using ONNX Runtime.
Falls back to PyTorch if the ONNX file is missing.

Usage:
    python fast_predict.py Bear1.png
    python fast_predict.py Bear1.png --onnx models/fast_cgh_net.onnx --output-bmp out.bmp
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Device resolution expected by the model / PLM
# ---------------------------------------------------------------------------
PLM_W, PLM_H = 1358, 800


def _load_image(image_path: str) -> np.ndarray:
    """Load image, convert to greyscale float32 [0,1], resize to PLM_H×PLM_W."""
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.shape != (PLM_H, PLM_W):
        arr = cv2.resize(arr, (PLM_W, PLM_H), interpolation=cv2.INTER_LINEAR)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr


# ---------------------------------------------------------------------------
# ONNX Runtime inference
# ---------------------------------------------------------------------------
def predict_onnx(image_path: str, onnx_path: str) -> np.ndarray:
    """Run a single image through the ONNX model and return the phase map."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    # Prefer CUDA if present, otherwise CPU
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    active = [p for p in preferred if p in providers]
    print(f"  ORT providers available : {providers}")
    print(f"  Using                   : {active}")

    sess = ort.InferenceSession(onnx_path, providers=active)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    arr = _load_image(image_path)                           # (H, W)
    x   = arr[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,H,W)

    t0     = time.perf_counter()
    result = sess.run([out_name], {inp_name: x})[0]
    elapsed = time.perf_counter() - t0

    phase = result.squeeze()                                # (H, W)
    print(f"  ORT inference time      : {elapsed * 1000:.1f} ms")
    return phase


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------
def predict_pytorch(image_path: str, pt_path: str) -> np.ndarray:
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from FastCGHNet import FastCGHNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FastCGHNet().to(device)
    ckpt   = torch.load(pt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  PyTorch device          : {device}")

    arr = _load_image(image_path)
    x   = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :]).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        phase_t = model(x)
    elapsed = time.perf_counter() - t0

    phase = phase_t.squeeze().cpu().numpy()
    print(f"  PyTorch inference time  : {elapsed * 1000:.1f} ms")
    return phase


# ---------------------------------------------------------------------------
# Optional: map phase → device BMP via TIPLMSuiteHologram helpers
# ---------------------------------------------------------------------------
def _phase_to_bmp(phase: np.ndarray, output_bmp: str) -> None:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from TIPLMSuiteHologram import DeviceLibrary, CGHGenerator
        device_lib  = DeviceLibrary()
        device_dict = device_lib.defineDevice("0.67")
        phase_disc, state_disc = CGHGenerator.discretePhase(
            phase, device_dict["nLevel"], device_dict["pLevel"]
        )
        cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
        cgh_u8     = cv2.normalize(cgh_mapped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(output_bmp, cgh_u8)
        print(f"  Saved device BMP        : {output_bmp}")
    except Exception as e:
        # Fallback: just save the normalised phase as a grey PNG
        fallback = str(output_bmp).replace(".bmp", "_phase.png")
        phase_u8 = ((phase - phase.min()) / (phase.max() - phase.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(fallback, phase_u8)
        print(f"  Device map failed ({e}); saved raw phase: {fallback}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FastCGHNet ONNX inference")
    parser.add_argument("image",                              help="Input image (PNG/JPG/BMP)")
    parser.add_argument("--onnx",   default="models/fast_cgh_net.onnx",
                        help="ONNX model path (default: models/fast_cgh_net.onnx)")
    parser.add_argument("--model",  default="models/best_model.pt",
                        help="PyTorch fallback model (used only when ONNX is missing)")
    parser.add_argument("--output-bmp",   dest="output_bmp",   help="Save CGH as BMP")
    parser.add_argument("--output-phase", dest="output_phase", help="Save raw phase as .npy")
    args = parser.parse_args()

    image_path = args.image
    onnx_path  = Path(args.onnx)
    pt_path    = Path(args.model)

    print(f"\nFastCGHNet ONNX inference")
    print(f"  Image : {image_path}")

    # ---- choose backend ------------------------------------------------
    if onnx_path.exists():
        print(f"  Model : {onnx_path}  [ONNX Runtime]")
        phase = predict_onnx(image_path, str(onnx_path))
    elif pt_path.exists():
        print(f"  ONNX not found at {onnx_path}; falling back to PyTorch: {pt_path}")
        phase = predict_pytorch(image_path, str(pt_path))
    else:
        sys.exit(f"No model found. Expected ONNX at '{onnx_path}' or PyTorch at '{pt_path}'.\n"
                 "Export ONNX with: python FastCGHNet.py export")

    print(f"  Phase range             : [{phase.min():.4f}, {phase.max():.4f}]")
    print(f"  Phase shape             : {phase.shape}")

    # ---- save outputs --------------------------------------------------
    if args.output_bmp:
        _phase_to_bmp(phase, args.output_bmp)

    if args.output_phase:
        np.save(args.output_phase, phase)
        print(f"  Saved phase .npy        : {args.output_phase}")

    if not args.output_bmp and not args.output_phase:
        # Default: save a quick preview PNG next to the input image
        preview = Path(image_path).stem + "_phase_preview.png"
        phase_u8 = ((phase - phase.min()) / (phase.max() - phase.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(preview, phase_u8)
        print(f"  Saved preview           : {preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
