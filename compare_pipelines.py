"""
compare_pipelines.py
====================
Runs Bear1.png through the ONNX (FastCGHNet) pipeline and
Bear2.png through the classic PLM.py iterative pipeline,
then produces a side-by-side difference report.

Usage:
    conda run -n Hologram python compare_pipelines.py
"""

import sys, time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PLM_W, PLM_H = 1358, 800
OUT = Path("comparison_output")
OUT.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (PLM_W, PLM_H), interpolation=cv2.INTER_AREA)

def norm_u8(arr):
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    return ((a - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

def ssim_simple(a, b):
    a, b = a.astype(np.float32)/255, b.astype(np.float32)/255
    mu_a = cv2.GaussianBlur(a,(11,11),1.5)
    mu_b = cv2.GaussianBlur(b,(11,11),1.5)
    s2_a = cv2.GaussianBlur(a*a,(11,11),1.5) - mu_a**2
    s2_b = cv2.GaussianBlur(b*b,(11,11),1.5) - mu_b**2
    s_ab = cv2.GaussianBlur(a*b,(11,11),1.5) - mu_a*mu_b
    c1,c2 = 0.01**2, 0.03**2
    return float(np.mean((2*mu_a*mu_b+c1)*(2*s_ab+c2) /
                         ((mu_a**2+mu_b**2+c1)*(s2_a+s2_b+c2)+1e-10)))

# ── Pipeline A: PyTorch (FastCGHNet .pt) ─────────────────────────────────────

def run_pt(image_path, pt_path="models/best_model.pt"):
    import torch
    from FastCGHNet import FastCGHNet

    ckpt   = torch.load(pt_path, map_location="cpu")
    model  = FastCGHNet(lite=ckpt.get("lite", False))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"    Epoch {ckpt.get('epoch')}  loss {ckpt.get('loss', '?'):.4f}")

    raw = load_gray(image_path).astype(np.float32) / 255.0
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    x   = torch.from_numpy(raw[np.newaxis, np.newaxis])

    t0 = time.perf_counter()
    with torch.no_grad():
        phase = model(x).squeeze().numpy()
    ms = (time.perf_counter() - t0) * 1000

    from PLM import DeviceLibrary, CGHGenerator
    D  = DeviceLibrary()
    G  = CGHGenerator()
    dd = D.defineDevice("0.67")
    _, state = G.discretePhase(phase, dd["nLevel"], dd["pLevel"])
    bmp = D.formatPLM(dd, state)
    return norm_u8(bmp), phase, ms

# ── Pipeline B: Classic iterative (PLM.py / ADAM+GS) ─────────────────────────

def run_plm(image_path, num_iter=200):
    from PLM import DeviceLibrary, CGHGenerator

    D  = DeviceLibrary()
    G  = CGHGenerator()
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
    return norm_u8(bmp), phase, ms

# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 62)
print("PIPELINE COMPARISON")
print("=" * 62)

# --- A: PyTorch on Bear1 -----------------------------------------------------
print("\n[A] Bear1.png  →  FastCGHNet (.pt) …")
try:
    bmp_a1, phase_a1, ms_a1 = run_pt("Bear1.png")
    cv2.imwrite(str(OUT / "Bear1_PT.bmp"), bmp_a1)
    print(f"    Time : {ms_a1:.0f} ms")
    print(f"    Saved: {OUT}/Bear1_PT.bmp")
    onnx1_ok = True
except Exception as e:
    print(f"    ❌  {e}")
    onnx1_ok = False

# --- A2: PyTorch on Bear2 ----------------------------------------------------
print("\n[A2] Bear2.png  →  FastCGHNet (.pt) …")
try:
    bmp_a2, phase_a2, ms_a2 = run_pt("Bear2.png")
    cv2.imwrite(str(OUT / "Bear2_PT.bmp"), bmp_a2)
    print(f"    Time : {ms_a2:.0f} ms")
    print(f"    Saved: {OUT}/Bear2_PT.bmp")
    onnx2_ok = True
except Exception as e:
    print(f"    ❌  {e}")
    onnx2_ok = False

# --- B: PLM iterative on Bear2 -----------------------------------------------
print("\n[B] Bear2.png  →  PLM.py iterative (200 iter) …")
try:
    bmp_b, phase_b, ms_b = run_plm("Bear2.png", num_iter=200)
    cv2.imwrite(str(OUT / "Bear2_PLM.bmp"), bmp_b)
    print(f"    Time : {ms_b:.0f} ms")
    print(f"    Saved: {OUT}/Bear2_PLM.bmp")
    plm_ok = True
except Exception as e:
    print(f"    ❌  {e}")
    plm_ok = False

# --- Diff & metrics ----------------------------------------------------------
def compare(name_a, img_a, name_b, img_b):
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    a, b = img_a[:h, :w], img_b[:h, :w]
    mse_v  = float(np.mean((a.astype(np.float32) - b.astype(np.float32))**2))
    psnr_v = 10 * np.log10(255**2 / (mse_v + 1e-10))
    ssim_v = ssim_simple(a, b)
    print(f"\n  {name_a}  vs  {name_b}")
    print(f"    MSE  : {mse_v:.1f}  |  PSNR: {psnr_v:.1f} dB  |  SSIM: {ssim_v:.4f}")
    return a, b, mse_v, ssim_v

print("\n" + "=" * 62)
print("DIFFERENCE METRICS")
print("=" * 62)

if onnx2_ok and plm_ok:
    a2, b, _, _ = compare("Bear2 PT", bmp_a2, "Bear2 PLM", bmp_b)
    diff = np.abs(a2.astype(np.float32) - b.astype(np.float32))
    cv2.imwrite(str(OUT / "Bear2_PT_vs_PLM_diff.png"),
                cv2.applyColorMap(norm_u8(diff), cv2.COLORMAP_INFERNO))

if onnx1_ok and onnx2_ok:
    compare("Bear1 PT", bmp_a1, "Bear2 PT", bmp_a2)

# 4-panel: Bear1-PT | Bear2-PT | Bear2-PLM | diff(PT vs PLM)
imgs, labels = [], []
if onnx1_ok: imgs.append(cv2.cvtColor(bmp_a1, cv2.COLOR_GRAY2BGR)); labels.append("Bear1 PT")
if onnx2_ok: imgs.append(cv2.cvtColor(bmp_a2, cv2.COLOR_GRAY2BGR)); labels.append("Bear2 PT")
if plm_ok:   imgs.append(cv2.cvtColor(bmp_b,  cv2.COLOR_GRAY2BGR)); labels.append("Bear2 PLM")
if onnx2_ok and plm_ok:
    diff_col = cv2.applyColorMap(norm_u8(diff), cv2.COLORMAP_INFERNO)
    imgs.append(diff_col); labels.append("Diff PT-PLM")

if imgs:
    panel = cv2.hconcat(imgs)
    scale = 300 / panel.shape[0]
    panel_s = cv2.resize(panel, (int(panel.shape[1]*scale), 300))
    n = len(labels)
    for i, lbl in enumerate(labels):
        x_off = int(i * panel_s.shape[1] / n) + 6
        cv2.putText(panel_s, lbl, (x_off, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "panel.png"), panel_s)
    print(f"\n  Saved: {OUT}/panel.png")

print("\nDone.")
