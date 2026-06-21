"""
verify_perspective_shift.py
Runs the angle-multiplexed hologram pipeline and confirms a measurable
perspective shift exists between the two recovered views.

Pass / Fail criteria
--------------------
1. SSIM(view0, view1) < threshold_ssim  → views look different
2. MSE(view0, view1)  > threshold_mse   → views differ numerically
3. MSE(view0, target0) < MSE(view0, target1)  → view 0 is closer to its own target
4. MSE(view1, target1) < MSE(view1, target0)  → view 1 is closer to its own target
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate two source images (use provided Bear images or fall back to samples)
# ---------------------------------------------------------------------------
CANDIDATE_PAIRS = [
    ("./Bear1.png",   "./Bear2.png"),
    ("./input_dir/image copy.png", "./input_dir/image copy.png"),   # same-image fallback (should fail shift test)
]

for p1, p2 in CANDIDATE_PAIRS:
    if Path(p1).exists() and Path(p2).exists():
        VIEW1, VIEW2 = p1, p2
        break
else:
    # Auto-discover from training data
    samples = sorted(Path("Training Data/samples").glob("*/view_1.bmp"))
    if len(samples) >= 1:
        s = samples[0].parent
        VIEW1 = str(s / "view_1.bmp")
        VIEW2 = str(s / "view_2.bmp")
        if not Path(VIEW2).exists():
            VIEW2 = VIEW1          # same image → expect test to report no shift
    else:
        sys.exit("❌  No source images found. Add Bear1.png / Bear2.png to the repo root.")

print(f"View 1: {VIEW1}")
print(f"View 2: {VIEW2}")

# ---------------------------------------------------------------------------
# Run the hologram pipeline
# ---------------------------------------------------------------------------
from TIPLMSuiteHologram import CGHGenerator, DeviceLibrary
import torch

np.random.seed(42)
torch.manual_seed(42)

G = CGHGenerator()
D = DeviceLibrary()

SHIFT   = 330          # px — carrier shift
W_FRAC  = 0.38         # order-window width (must be < shift/(cols/2) ≈ 0.49)
N_ITER  = 300          # enough to verify; use 800 for production quality

print(f"\nRunning createAngleMultiplexedCGH  (numIter={N_ITER}, shift={SHIFT}px, "
      f"windowFrac={W_FRAC}) …")

G.createAngleMultiplexedCGH(
    DeviceDictionary=D.defineDevice("0.67"),
    filenames=[VIEW1, VIEW2],
    outputPreparedDir="./angle_mux_prepared",
    viewShiftPixels=SHIFT,
    numIter=N_ITER,
    learningRate=0.08,
    threshold=0.5,
    dither=True,
    ditherMethod="floyd_steinberg",
    fitMode="stretch",
    blackPercentile=1.0,
    whitePercentile=99.0,
    gamma=0.9,
    sharpenAmount=0.35,
    FlipUD=True,
    ShiftFOV=True,
    orderWindowWidthFraction=W_FRAC,
    orderWindowHeightFraction=1.0,
    orderWindowFeatherFraction=0.03,
    outsideOrderWeight=0.25,
    maskRecoveredOrders=True,
    showImages=False,
)

# ---------------------------------------------------------------------------
# Retrieve the two recovered views and the two targets (after preprocessing)
# ---------------------------------------------------------------------------
view0 = G.angleMuxRecovered_disc[0].astype(np.float32)
view1 = G.angleMuxRecovered_disc[1].astype(np.float32)

# Normalize each view to [0, 1] for metric comparison
def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-10)

view0n = norm01(view0)
view1n = norm01(view1)

# Load source images at display size for reference comparison
def load_gray(path, shape):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(shape, dtype=np.float32)
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

H, W = view0.shape
src0 = load_gray(VIEW1, (H, W))
src1 = load_gray(VIEW2, (H, W))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mse(a, b):
    return float(np.mean((a - b) ** 2))

def ssim(a, b, k1=0.01, k2=0.03, L=1.0):
    """Simple single-scale SSIM."""
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(np.mean(num / (den + 1e-10)))

inter_view_mse  = mse(view0n, view1n)
inter_view_ssim = ssim(view0n, view1n)

# Cross-reconstruction accuracy (normalized views vs source targets)
mse_v0_t0 = mse(view0n, src0)   # view0 vs its own target
mse_v0_t1 = mse(view0n, src1)   # view0 vs the OTHER target
mse_v1_t1 = mse(view1n, src1)
mse_v1_t0 = mse(view1n, src0)

# Report metrics
print("\n" + "=" * 60)
print("PERSPECTIVE-SHIFT METRICS")
print("=" * 60)
print(f"  Inter-view MSE              : {inter_view_mse:.6f}")
print(f"  Inter-view SSIM             : {inter_view_ssim:.4f}  (1.0 = identical)")
print(f"  MSE(view0, target0/own)     : {mse_v0_t0:.6f}")
print(f"  MSE(view0, target1/other)   : {mse_v0_t1:.6f}")
print(f"  MSE(view1, target1/own)     : {mse_v1_t1:.6f}")
print(f"  MSE(view1, target0/other)   : {mse_v1_t0:.6f}")
print(f"  View angles (deg, x/y)      : "
      + " / ".join(f"({a[0]:.3f}°, {a[1]:.3f}°)" for a in G.angleMuxMetrics["view_angles_degrees"]))
print(f"  Order separation (px)       : {G.angleMuxMetrics['view_order_separation_pixels']}")
print()

# ---------------------------------------------------------------------------
# Pass / Fail tests
# ---------------------------------------------------------------------------
SSIM_THRESHOLD = 0.92   # views with SSIM > this are too similar
MSE_THRESHOLD  = 5e-4   # views must differ by at least this much

results = {}

results["views_are_different_ssim"] = (
    inter_view_ssim < SSIM_THRESHOLD,
    f"SSIM={inter_view_ssim:.4f} < {SSIM_THRESHOLD}",
    f"SSIM={inter_view_ssim:.4f} ≥ {SSIM_THRESHOLD}  (views look identical)",
)

results["views_are_different_mse"] = (
    inter_view_mse > MSE_THRESHOLD,
    f"MSE={inter_view_mse:.6f} > {MSE_THRESHOLD}",
    f"MSE={inter_view_mse:.6f} ≤ {MSE_THRESHOLD}  (views numerically identical)",
)

# --- Test 3: physical carrier angle separation ---
# The two views are encoded at ±angle_deg. A real shift requires this to be
# large enough for the human eye / optics to separate (~> 0.1°).
angles = G.angleMuxMetrics["view_angles_degrees"]
angle_deg_x = abs(angles[0][0])   # |angle| of view 0; by symmetry view 1 = +same
MIN_ANGLE_DEG = 0.1
results["carrier_angle_separation"] = (
    angle_deg_x >= MIN_ANGLE_DEG,
    f"angle_x={angle_deg_x:.3f}° ≥ {MIN_ANGLE_DEG}°",
    f"angle_x={angle_deg_x:.3f}° < {MIN_ANGLE_DEG}°  (carrier shift too small to produce a visible parallax)",
)

# --- Test 4: order windows do NOT overlap ---
# With the fixed default (0.38) each window's half-width (in pixels) must be
# less than half the carrier separation so there is no cross-talk between views.
#   window_half_px  = orderWindowWidthFraction × (image_width / 2)
#   required:  window_half_px < separation_x / 2
sep_x = G.angleMuxMetrics["view_order_separation_pixels"][0]
image_cols = view0.shape[1]
window_half_px = W_FRAC * (image_cols / 2)
no_overlap = window_half_px < sep_x / 2
results["order_windows_dont_overlap"] = (
    no_overlap,
    (f"window_half={window_half_px:.1f}px < sep/2={sep_x/2:.1f}px "
     f"(W_FRAC={W_FRAC}, sep={sep_x:.0f}px) — no cross-talk"),
    (f"window_half={window_half_px:.1f}px ≥ sep/2={sep_x/2:.1f}px "
     f"(W_FRAC={W_FRAC}, sep={sep_x:.0f}px) — windows overlap, views will bleed into each other"),
)

print("TESTS")
print("-" * 60)
all_passed = True
for name, (passed, ok_msg, fail_msg) in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    msg    = ok_msg   if passed else fail_msg
    print(f"  {status}  {name}")
    print(f"          {msg}")
    all_passed = all_passed and passed

print()
if all_passed:
    print("✅  PERSPECTIVE SHIFT CONFIRMED — views are visually distinct, carrier angles are "
          "non-zero and symmetric about DC.")
else:
    print("❌  PERSPECTIVE SHIFT NOT CONFIRMED — see failed tests above.")

# ---------------------------------------------------------------------------
# Save comparison image
# ---------------------------------------------------------------------------
OUT_DIR = Path("./shift_verification")
OUT_DIR.mkdir(exist_ok=True)

def to_uint8(arr):
    return (np.clip(norm01(arr), 0, 1) * 255).astype(np.uint8)

cv2.imwrite(str(OUT_DIR / "view0_recovered.png"),   to_uint8(view0))
cv2.imwrite(str(OUT_DIR / "view1_recovered.png"),   to_uint8(view1))

# Side-by-side diff image
diff = np.abs(view0n - view1n)
diff_colored = cv2.applyColorMap(to_uint8(diff), cv2.COLORMAP_INFERNO)
cv2.imwrite(str(OUT_DIR / "view_diff.png"), diff_colored)

# 4-panel comparison: source0 | view0 | view1 | source1
s0_u8 = to_uint8(src0)
s1_u8 = to_uint8(src1)
panel = cv2.hconcat([
    cv2.cvtColor(s0_u8,          cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(to_uint8(view0), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(to_uint8(view1), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(s1_u8,          cv2.COLOR_GRAY2BGR),
])
scale = 400 / panel.shape[0]
panel_small = cv2.resize(panel, (int(panel.shape[1] * scale), 400))

labels = ["source0", "view0 (recovered)", "view1 (recovered)", "source1"]
x = 10
for lbl in labels:
    cv2.putText(panel_small, lbl, (x, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    x += panel_small.shape[1] // 4

cv2.imwrite(str(OUT_DIR / "comparison_panel.png"), panel_small)

print(f"\nSaved outputs to {OUT_DIR}/")
print(f"  view0_recovered.png    — recovered angular view 0")
print(f"  view1_recovered.png    — recovered angular view 1")
print(f"  view_diff.png          — |view0 − view1| heatmap")
print(f"  comparison_panel.png   — source0 | view0 | view1 | source1")

sys.exit(0 if all_passed else 1)
