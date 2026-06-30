#!/usr/bin/env python3
"""
benchmark_cnn_vs_plm.py
=======================
Comprehensive benchmark comparing FastCGHNet (CNN) vs. classic PLM (ADAM+GS).
Runs both approaches on Bear1 and Bear2, measures quality and speed degradation.

Usage:
    python benchmark_cnn_vs_plm.py
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PLM_W, PLM_H = 1358, 800
OUT = Path("benchmark_output")
OUT.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_gray(path):
    """Load and normalize grayscale image to PLM resolution."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (PLM_W, PLM_H), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def norm_u8(arr):
    """Normalize array to uint8 [0,255]."""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    return ((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

def ssim_simple(a, b):
    """Compute SSIM between two images."""
    a, b = a.astype(np.float32)/255, b.astype(np.float32)/255
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    s2_a = cv2.GaussianBlur(a*a, (11, 11), 1.5) - mu_a**2
    s2_b = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mu_b**2
    s_ab = cv2.GaussianBlur(a*b, (11, 11), 1.5) - mu_a*mu_b
    c1, c2 = 0.01**2, 0.03**2
    ssim_val = (2*mu_a*mu_b+c1)*(2*s_ab+c2) / ((mu_a**2+mu_b**2+c1)*(s2_a+s2_b+c2)+1e-10)
    return float(np.mean(ssim_val))

def reconstruct_image_from_phase(phase_rad, target_shape=(PLM_H, PLM_W)):
    """
    Simulate what image would be reconstructed from a hologram phase.
    Uses the same Fourier model as the CNN loss function.
    """
    field = np.exp(1j * phase_rad)
    recon = np.fft.ifft2(field)
    intensity = np.real(recon)**2 + np.imag(recon)**2
    
    # Shift FOV (same as in loss function) - use shift instead of shifts for numpy compatibility
    intensity = np.roll(intensity, intensity.shape[0]//2, axis=0)
    intensity = np.roll(intensity, intensity.shape[1]//2, axis=1)
    
    # Normalize to [0, 1]
    intensity = np.clip(intensity, 0, 1)
    return intensity

def psnr_from_phase(predicted_phase, target_img):
    """
    Compute PSNR by reconstructing the image from predicted phase and
    comparing to target. This is the metric used during training.
    """
    recon = reconstruct_image_from_phase(predicted_phase)
    
    # Compute optimal brightness gain (same as loss)
    num = np.mean(recon * target_img)
    den = np.mean(recon * recon) + 1e-20
    s = num / den
    
    mse = np.mean((s * recon - target_img) ** 2)
    psnr_val = -10.0 * np.log10(mse + 1e-12)
    return psnr_val, mse, s * recon

# ────────────────────────────────────────────────────────────────────────────
# Pipeline A: FastCGHNet (CNN)
# ────────────────────────────────────────────────────────────────────────────

def run_cnn(image_path, pt_path="models/best_model.pt"):
    """Run FastCGHNet CNN inference."""
    import torch
    from FastCGHNet import FastCGHNet
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(pt_path, map_location=device)
    model = FastCGHNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch_info = checkpoint.get('epoch', '?')
    loss_info = checkpoint.get('loss', '?')
    
    raw = load_gray(image_path)
    x = torch.from_numpy(raw[np.newaxis, np.newaxis, :, :]).to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        phase = model(x).squeeze().cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000
    
    from PLM import DeviceLibrary, CGHGenerator
    D = DeviceLibrary()
    G = CGHGenerator()
    dd = D.defineDevice("0.67")
    _, state = G.discretePhase(phase, dd["nLevel"], dd["pLevel"])
    bmp = D.formatPLM(dd, state)
    
    return {
        'phase': phase,
        'bmp': norm_u8(bmp),
        'time_ms': ms,
        'device': device.type,
        'epoch': epoch_info,
        'loss': loss_info,
    }

# ────────────────────────────────────────────────────────────────────────────
# Pipeline B: Classic PLM (ADAM+GS)
# ────────────────────────────────────────────────────────────────────────────

def run_plm(image_path, num_iter=200):
    """Run classic iterative PLM approach."""
    from PLM import DeviceLibrary, CGHGenerator
    
    D = DeviceLibrary()
    G = CGHGenerator()
    dd = D.defineDevice("0.67")
    
    t0 = time.perf_counter()
    G.createCGH(
        dd,
        filename=str(image_path),
        alg="ADAMWGS",
        numIter=num_iter,
        initialPhase="Random",
        propMethod="Fourier",
        ShiftFOV=True,
        FlipUD=True,
        showImages=False,
        lossMode="mse",
    )
    ms = (time.perf_counter() - t0) * 1000
    
    bmp = G.CGH_mapped
    phase = G.CGH_output_cont
    
    return {
        'phase': phase,
        'bmp': norm_u8(bmp),
        'time_ms': ms,
        'num_iter': num_iter,
    }

# ────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("CNN vs. CLASSIC PLM BENCHMARK")
print("=" * 80)

results = {}

for img_name in ["Bear1.png", "Bear2.png"]:
    print(f"\n{'─' * 80}")
    print(f"Processing: {img_name}")
    print(f"{'─' * 80}")
    
    if not Path(img_name).exists():
        print(f"  ⚠️  {img_name} not found, skipping")
        continue
    
    target_img = load_gray(img_name)
    
    # ── CNN ──
    print(f"\n  [CNN] FastCGHNet inference…")
    try:
        cnn_result = run_cnn(img_name)
        print(f"       ✓ Inference time: {cnn_result['time_ms']:.1f} ms")
        print(f"       ✓ Device: {cnn_result['device']}")
        print(f"       ✓ Model epoch: {cnn_result['epoch']}, loss: {cnn_result['loss']}")
        
        cnn_psnr, cnn_mse, cnn_recon = psnr_from_phase(cnn_result['phase'], target_img)
        print(f"       ✓ Reconstruction PSNR: {cnn_psnr:.2f} dB")
        
        cv2.imwrite(str(OUT / f"{img_name.split('.')[0]}_CNN_bmp.png"), cnn_result['bmp'])
        results[f"{img_name}_cnn"] = cnn_result
        cnn_ok = True
    except Exception as e:
        print(f"       ✗ Error: {e}")
        cnn_ok = False
    
    # ── PLM (classic) ──
    print(f"\n  [PLM] Classic ADAM+GS iterative (200 iterations)…")
    try:
        plm_result = run_plm(img_name, num_iter=200)
        print(f"       ✓ Inference time: {plm_result['time_ms']:.1f} ms")
        
        plm_psnr, plm_mse, plm_recon = psnr_from_phase(plm_result['phase'], target_img)
        print(f"       ✓ Reconstruction PSNR: {plm_psnr:.2f} dB")
        
        cv2.imwrite(str(OUT / f"{img_name.split('.')[0]}_PLM_bmp.png"), plm_result['bmp'])
        results[f"{img_name}_plm"] = plm_result
        plm_ok = True
    except Exception as e:
        print(f"       ✗ Error: {e}")
        plm_ok = False
    
    # ── Compare ──
    if cnn_ok and plm_ok:
        print(f"\n  [COMPARISON]")
        time_ratio = cnn_result['time_ms'] / plm_result['time_ms']
        psnr_diff = cnn_psnr - plm_psnr
        
        print(f"       Speed: CNN={cnn_result['time_ms']:.1f}ms vs PLM={plm_result['time_ms']:.1f}ms")
        print(f"       Speed ratio: {time_ratio:.2f}x (CNN is {time_ratio:.1f}x {'faster' if time_ratio < 1 else 'slower'})")
        print(f"       Quality: CNN PSNR={cnn_psnr:.2f}dB vs PLM PSNR={plm_psnr:.2f}dB")
        print(f"       Quality diff: {psnr_diff:+.2f} dB (CNN is {abs(psnr_diff):.2f}dB {'better' if psnr_diff > 0 else 'worse'})")
        
        # BMP comparison
        bmp_mse = float(np.mean((cnn_result['bmp'].astype(np.float32) - 
                                 plm_result['bmp'].astype(np.float32))**2))
        bmp_ssim = ssim_simple(cnn_result['bmp'], plm_result['bmp'])
        
        print(f"       BMP diff: MSE={bmp_mse:.1f}, SSIM={bmp_ssim:.4f}")
        
        # Save diff visualization
        diff_map = np.abs(cnn_result['bmp'].astype(np.float32) - 
                         plm_result['bmp'].astype(np.float32))
        diff_color = cv2.applyColorMap(norm_u8(diff_map), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(OUT / f"{img_name.split('.')[0]}_diff.png"), diff_color)
        print(f"       ✓ Saved difference heatmap: {img_name.split('.')[0]}_diff.png")

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

if "Bear1.png_cnn" in results and "Bear2.png_plm" in results:
    print(f"\nBear1 (CNN only): {results['Bear1.png_cnn']['time_ms']:.1f} ms inference")

if "Bear2.png_cnn" in results and "Bear2.png_plm" in results:
    cnn_r = results['Bear2.png_cnn']
    plm_r = results['Bear2.png_plm']
    target = load_gray("Bear2.png")
    
    cnn_psnr, _, _ = psnr_from_phase(cnn_r['phase'], target)
    plm_psnr, _, _ = psnr_from_phase(plm_r['phase'], target)
    
    print(f"\nBear2 (CNN vs PLM):")
    print(f"  Speed:      CNN {cnn_r['time_ms']:.1f}ms vs PLM {plm_r['time_ms']:.1f}ms")
    print(f"              → CNN is {plm_r['time_ms']/cnn_r['time_ms']:.0f}x faster")
    print(f"  Quality:    CNN {cnn_psnr:.2f}dB vs PLM {plm_psnr:.2f}dB")
    print(f"              → CNN degrades by {plm_psnr - cnn_psnr:.2f} dB")
    print(f"  Trade-off:  {plm_r['time_ms']/cnn_r['time_ms']:.0f}x speedup for {abs(plm_psnr - cnn_psnr):.2f}dB quality loss")

print(f"\nAll outputs saved to: {OUT}/")
print("\nDone.")
