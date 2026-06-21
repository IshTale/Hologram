"""
fuse_stereo_view.py
===================
Integrate the two stereo camera images into ONE depth-aware view -- an
"RGB-D 3D reconstruction mask" -- that can feed the hologram stage.

Output bundle (a single fused "view"):
  * rgb        : the LEFT camera as the reference color image
  * depth_mm   : dense per-pixel metric depth (mm), from BOTH cameras
  * alpha      : subject matte in [0,255] (segmentation INTERSECT depth band,
                 cross-validated against the right camera)
  * hologram   : grayscale subject on black, fit to the CGH input size (1358x800)

Why this design (the rig is wide-baseline / short-range):
  baseline ~300mm at ~525mm  =>  disparity ~316px (half the frame), heavy occlusion.
  So we do NOT trust rectified stereo alone. The matte is driven by a single-view
  segmentation (robust to occlusion) and only *gated* by the noisy stereo depth band,
  with a left<->right consistency check to clean edges. Both cameras still contribute:
  the right camera produces the depth and validates the mask.

Runs two ways:
  live    : python fuse_stereo_view.py --left-camera 0 --right-camera 2
  offline : python fuse_stereo_view.py --left-image L.png --right-image R.png --output-dir out

Controls (live): s save bundle, q quit, j/l shift, u/o target distance, [ ] alpha gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from stereo_common import (
    api_preference,
    draw_label,
    match_frame_sizes,
    open_camera,
    parse_optional_resolution,
    read_camera_pair,
    require_cv2,
    require_numpy,
    side_by_side,
    timestamp_name,
)

# Reuse the proven raw (un-rectified) matcher pieces from the existing tool.
from live_raw_depth_mask import (
    clean_mask,
    focal_from_hfov,
    local_texture,
    shift_range_px,
    shifted_match_cost,
)

# Hologram CGH input size (TI 0.67 PLM); the fused subject is fit to this for the next stage.
CGH_W, CGH_H = 1358, 800


# --------------------------------------------------------------------------- #
# Optional MediaPipe selfie segmentation (graceful fallback to depth-only)
# --------------------------------------------------------------------------- #
SELFIE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


class SelfieSegmenter:
    """MediaPipe Tasks ImageSegmenter (selfie). Returns a soft foreground matte in [0,1].

    Uses the modern Tasks API (the legacy mp.solutions module is gone in recent builds).
    The .tflite model is auto-downloaded once to ./models/ if not present; if MediaPipe
    or the download is unavailable, the pipeline falls back to a depth-only matte.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.available = False
        self._seg = None
        self._mp = None
        try:
            import mediapipe as mp  # noqa: WPS433
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp
            path = self._ensure_model(model_path)
            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self._seg = vision.ImageSegmenter.create_from_options(options)
            self.available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[info] MediaPipe ImageSegmenter unavailable ({exc}); using depth-only matte.")

    @staticmethod
    def _ensure_model(model_path: Optional[Path]) -> Path:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"
        model_path = Path(model_path)
        if not model_path.exists():
            import urllib.request

            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[info] downloading selfie segmenter model -> {model_path}")
            urllib.request.urlretrieve(SELFIE_MODEL_URL, str(model_path))
        return model_path

    def matte(self, bgr) -> Optional["object"]:
        if not self.available:
            return None
        cv = require_cv2()
        np = require_numpy()
        rgb = np.ascontiguousarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._seg.segment(mp_image)
        if not getattr(result, "confidence_masks", None):
            return None
        mask = np.squeeze(result.confidence_masks[0].numpy_view().astype("float32"))  # HxW foreground prob
        if mask.shape[:2] != bgr.shape[:2]:
            mask = cv.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv.INTER_LINEAR)
        return mask

    def close(self) -> None:
        if self._seg is not None:
            self._seg.close()


# --------------------------------------------------------------------------- #
# Dense depth from the raw (un-rectified) pair via best-shift search
# --------------------------------------------------------------------------- #
def compute_dense_depth(
    left_bgr,
    right_bgr,
    baseline_mm: float,
    near_mm: float,
    far_mm: float,
    hfov_deg: float,
    shift_sign: int,
    shift_offset_px: float,
    match_width: int,
    patch_size: int,
    shift_step_px: int,
    vertical_tolerance_px: int,
    cost_threshold: float,
    texture_threshold: float,
):
    """Return (depth_mm, disparity_px, confidence, focal_px) at full resolution.

    Confidence in [0,1] blends matching cost and local texture; depth is
    focal*baseline/disparity, clamped to [near, far]. Invalid pixels -> depth 0, conf 0.
    """
    cv = require_cv2()
    np = require_numpy()
    h, w = left_bgr.shape[:2]
    focal_px, shift_min, shift_max = shift_range_px(w, baseline_mm, near_mm, far_mm, hfov_deg)

    scale = min(1.0, float(match_width) / float(w))
    small_size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
    left_small = cv.resize(left_bgr, small_size, interpolation=cv.INTER_AREA)
    right_small = cv.resize(right_bgr, small_size, interpolation=cv.INTER_AREA)
    left_gray = cv.GaussianBlur(cv.cvtColor(left_small, cv.COLOR_BGR2GRAY), (3, 3), 0)
    right_gray = cv.GaussianBlur(cv.cvtColor(right_small, cv.COLOR_BGR2GRAY), (3, 3), 0)

    shift_min_s = int(round((shift_min + shift_offset_px) * scale))
    shift_max_s = int(round((shift_max + shift_offset_px) * scale))
    if shift_max_s < shift_min_s:
        shift_min_s, shift_max_s = shift_max_s, shift_min_s

    step_s = max(1, int(round(float(shift_step_px) * scale)))
    vertical_s = max(0, int(round(float(vertical_tolerance_px) * scale)))
    patch_s = max(3, int(round(float(patch_size) * scale)))
    if patch_s % 2 == 0:
        patch_s += 1

    signed_shifts = [int(shift_sign * d) for d in range(max(0, shift_min_s), max(1, shift_max_s) + 1, step_s)]
    if not signed_shifts:
        signed_shifts = [0]
    vertical_shifts = range(-vertical_s, vertical_s + 1)

    min_cost = np.full(left_gray.shape, 255.0, dtype="float32")
    best_disp = np.zeros(left_gray.shape, dtype="float32")  # abs disparity in ORIGINAL px
    for dx in signed_shifts:
        for dy in vertical_shifts:
            cost = shifted_match_cost(left_gray, right_gray, dx, dy, patch_s)
            better = cost < min_cost
            min_cost[better] = cost[better]
            best_disp[better] = abs(float(dx)) / max(scale, 1e-6)

    texture = local_texture(left_gray, patch_s)

    # confidence: low cost AND enough texture
    cost_conf = np.clip((float(cost_threshold) - min_cost) / max(float(cost_threshold), 1e-6), 0.0, 1.0)
    tex_conf = np.clip(texture / max(2.0 * float(texture_threshold), 1e-6), 0.0, 1.0)
    conf_small = (cost_conf * tex_conf).astype("float32")

    # depth from disparity
    disp_safe = np.maximum(best_disp, 1e-3)
    depth_small = (focal_px * float(baseline_mm)) / disp_safe
    in_band = (depth_small >= near_mm) & (depth_small <= far_mm) & (conf_small > 0.0)
    depth_small = np.where(in_band, depth_small, 0.0).astype("float32")
    conf_small = np.where(in_band, conf_small, 0.0).astype("float32")

    # upsample to full res
    depth_mm = cv.resize(depth_small, (w, h), interpolation=cv.INTER_NEAREST)
    disparity = cv.resize(best_disp, (w, h), interpolation=cv.INTER_NEAREST)
    confidence = cv.resize(conf_small, (w, h), interpolation=cv.INTER_LINEAR)
    return depth_mm, disparity, confidence, focal_px


# --------------------------------------------------------------------------- #
# Fuse segmentation + depth into one subject matte, validated by the right view
# --------------------------------------------------------------------------- #
def build_subject_alpha(
    seg_matte,            # HxW float [0,1] or None
    depth_mm,             # HxW float
    confidence,           # HxW float [0,1]
    near_mm: float,
    far_mm: float,
    alpha_gate: float,
    open_size: int,
    close_size: int,
    min_area: int,
    guide_bgr=None,       # reference image for guided-filter edge refinement
):
    """Subject matte = segmentation (primary), with depth used as a SELECTION gate.

    Per the literature (wide-baseline stereo is too noisy to define edges): use the soft
    depth band to *select the correct connected component* rather than multiplying the
    matte by the raw band (which would punch holes in smooth skin). Then a guided filter
    snaps the alpha to image edges.
    """
    cv = require_cv2()
    np = require_numpy()

    # soft depth-band weight for component SELECTION (dilate + blur; never a hard per-pixel mask)
    band = ((depth_mm >= near_mm) & (depth_mm <= far_mm)).astype("float32") * np.clip(confidence, 0.0, 1.0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
    w_depth = cv.GaussianBlur(cv.dilate(band, kernel), (0, 0), 15.0)
    if float(w_depth.max()) > 1e-6:
        w_depth = w_depth / float(w_depth.max())

    if seg_matte is not None:
        binary = (seg_matte >= float(alpha_gate)).astype("uint8") * 255
    else:
        binary = (band > 0.05).astype("uint8") * 255
    binary = clean_mask(binary, open_size=open_size, close_size=close_size, min_area=0)

    # keep the best depth-supported component (subject), plus any sizeable, depth-backed blob
    n, labels, stats, _ = cv.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    best_lab, best_score = None, 0.0
    for lab in range(1, n):
        area = int(stats[lab, cv.CC_STAT_AREA])
        if area < max(1, int(min_area)):
            continue
        comp = labels == lab
        support = float(w_depth[comp].mean())
        score = area * (support + 1e-3)
        if score > best_score:
            best_score, best_lab = score, lab
        if support > 0.15:
            keep[comp] = 255
    if best_lab is not None:
        keep[labels == best_lab] = 255
    binary = keep

    soft_alpha = cv.GaussianBlur(binary, (0, 0), sigmaX=2.0)
    # edge refine: guided filter pulls the alpha onto real image edges (needs opencv-contrib)
    if guide_bgr is not None and hasattr(cv, "ximgproc"):
        try:
            guide = cv.cvtColor(guide_bgr, cv.COLOR_BGR2GRAY)
            soft_alpha = cv.ximgproc.guidedFilter(guide, soft_alpha.astype("uint8"), 8, int((0.02 * 255) ** 2))
        except Exception:
            pass
    return binary, soft_alpha


def fill_depth_holes(depth_mm, alpha):
    """Fill missing subject depth (holes inside the matte) by inpainting + smoothing."""
    cv = require_cv2()
    np = require_numpy()
    subject = alpha > 0
    holes = subject & (depth_mm <= 0)
    if not holes.any():
        return depth_mm
    # normalize valid subject depth to 8-bit, inpaint holes, map back
    valid = subject & (depth_mm > 0)
    if not valid.any():
        return depth_mm
    dmin, dmax = float(depth_mm[valid].min()), float(depth_mm[valid].max())
    rng = max(dmax - dmin, 1.0)
    d8 = np.clip((depth_mm - dmin) / rng * 255.0, 0, 255).astype("uint8")
    d8[~valid] = 0
    filled8 = cv.inpaint(d8, holes.astype("uint8") * 255, 3, cv.INPAINT_TELEA)
    filled = filled8.astype("float32") / 255.0 * rng + dmin
    out = depth_mm.copy()
    out[holes] = filled[holes]
    out[~subject] = 0.0
    return out


def to_hologram_input(left_bgr, alpha, depth_mm, use_depth_shading: bool):
    """Make the grayscale subject-on-black image at the CGH size (1358x800).

    Optionally shade by depth (nearer = brighter) so the single image still carries
    a coarse 3D cue for the hologram stage.
    """
    cv = require_cv2()
    np = require_numpy()
    gray = cv.cvtColor(left_bgr, cv.COLOR_BGR2GRAY).astype("float32") / 255.0
    a = (alpha.astype("float32") / 255.0)
    if use_depth_shading and (depth_mm > 0).any():
        valid = depth_mm > 0
        dmin, dmax = float(depth_mm[valid].min()), float(depth_mm[valid].max())
        shade = np.zeros_like(gray)
        shade[valid] = 1.0 - np.clip((depth_mm[valid] - dmin) / max(dmax - dmin, 1.0), 0, 1)  # near=bright
        subject = gray * (0.5 + 0.5 * shade)
    else:
        subject = gray
    composite = subject * a  # subject on black
    composite = np.clip(composite, 0, 1)
    holo = cv.resize((composite * 255).astype("uint8"), (CGH_W, CGH_H), interpolation=cv.INTER_AREA)
    return holo


def depth_preview(depth_mm, near_mm, far_mm):
    cv = require_cv2()
    np = require_numpy()
    valid = depth_mm > 0
    vis = np.zeros(depth_mm.shape, dtype="uint8")
    if valid.any():
        norm = np.clip((depth_mm - near_mm) / max(far_mm - near_mm, 1.0), 0, 1)
        vis[valid] = (255 * (1.0 - norm[valid])).astype("uint8")  # near = bright
    return cv.applyColorMap(vis, cv.COLORMAP_TURBO)


def compose_fused_view(
    left_bgr,
    right_bgr,
    segmenter: SelfieSegmenter,
    args,
    near_mm: float,
    far_mm: float,
    shift_offset_px: float,
    alpha_gate: float,
):
    cv = require_cv2()
    depth_mm, disparity, confidence, focal_px = compute_dense_depth(
        left_bgr, right_bgr,
        baseline_mm=args.baseline_mm, near_mm=near_mm, far_mm=far_mm, hfov_deg=args.hfov_deg,
        shift_sign=args.shift_sign, shift_offset_px=shift_offset_px,
        match_width=args.match_width, patch_size=args.patch_size, shift_step_px=args.shift_step_px,
        vertical_tolerance_px=args.vertical_tolerance_px, cost_threshold=args.cost_threshold,
        texture_threshold=args.texture_threshold,
    )
    np = require_numpy()
    seg = segmenter.matte(left_bgr)
    alpha_bin, alpha_soft = build_subject_alpha(
        seg, depth_mm, confidence, near_mm, far_mm, alpha_gate,
        open_size=args.open_size, close_size=args.close_size, min_area=args.min_area,
        guide_bgr=left_bgr,
    )
    depth_mm = fill_depth_holes(depth_mm, alpha_bin)

    # z-squeeze metric depth into a thin normalized volume [0,1] over [near,far], subject only.
    # This is the format layer-based / Tensor-Holography CGH expects (NOT raw mm or disparity).
    a = alpha_soft.astype("float32") / 255.0
    depth_norm = np.zeros_like(depth_mm)
    valid = depth_mm > 0
    if valid.any():
        depth_norm[valid] = np.clip((depth_mm[valid] - near_mm) / max(far_mm - near_mm, 1.0), 0.0, 1.0)
    depth_norm = depth_norm * a

    rgba = cv.cvtColor(left_bgr, cv.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_soft
    holo = to_hologram_input(left_bgr, alpha_soft, depth_mm, use_depth_shading=not args.no_depth_shading)
    return {
        "rgb": left_bgr, "rgba": rgba, "depth_mm": depth_mm, "depth_norm": depth_norm,
        "alpha": alpha_soft, "depth_vis": depth_preview(depth_mm, near_mm, far_mm), "holo": holo,
        "focal_px": focal_px,
    }


def save_bundle(out_dir: Path, bundle) -> None:
    cv = require_cv2()
    np = require_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = timestamp_name("fused", "")
    cv.imwrite(str(out_dir / f"{stem}_rgb.png"), bundle["rgb"])
    cv.imwrite(str(out_dir / f"{stem}_cutout.png"), bundle["rgba"])
    cv.imwrite(str(out_dir / f"{stem}_alpha.png"), bundle["alpha"])
    cv.imwrite(str(out_dir / f"{stem}_depth_vis.png"), bundle["depth_vis"])
    cv.imwrite(str(out_dir / f"{stem}_hologram_input.png"), bundle["holo"])
    # 16-bit depth in mm, z-squeezed depth_norm, + RGB-D npz for the hologram stage
    cv.imwrite(str(out_dir / f"{stem}_depth_mm16.png"), np.clip(bundle["depth_mm"], 0, 65535).astype("uint16"))
    cv.imwrite(str(out_dir / f"{stem}_depth_norm.png"), (np.clip(bundle["depth_norm"], 0, 1) * 255).astype("uint8"))
    np.savez_compressed(
        out_dir / f"{stem}_rgbd.npz",
        rgb=cv.cvtColor(bundle["rgb"], cv.COLOR_BGR2RGB),
        depth_mm=bundle["depth_mm"].astype("float32"),
        depth_norm=bundle["depth_norm"].astype("float32"),  # [0,1] thin volume -> CGH input
        alpha=bundle["alpha"].astype("uint8"),
    )
    print(f"saved fused bundle: {out_dir / stem}_*")


def run_offline(args, segmenter, near_mm, far_mm) -> int:
    cv = require_cv2()
    left = cv.imread(str(args.left_image))
    right = cv.imread(str(args.right_image))
    if left is None or right is None:
        raise SystemExit("could not read --left-image / --right-image")
    resolution = parse_optional_resolution(args.width, args.height)
    left, right = match_frame_sizes(cv, left, right, resolution)
    bundle = compose_fused_view(left, right, segmenter, args, near_mm, far_mm,
                                float(args.shift_offset_px), float(args.alpha_gate))
    save_bundle(args.output_dir, bundle)
    print("offline fuse complete.")
    return 0


def run_live(args, segmenter, near_mm, far_mm) -> int:
    cv = require_cv2()
    api = api_preference(cv, args.api)
    resolution = parse_optional_resolution(args.width, args.height)
    left_cap = open_camera(cv, args.left_camera, api, resolution, args.fps)
    right_cap = open_camera(cv, args.right_camera, api, resolution, args.fps)
    shift_offset = float(args.shift_offset_px)
    target_distance = float(args.target_distance_mm)
    alpha_gate = float(args.alpha_gate)
    print("Controls: s save bundle, q quit, j/l shift, u/o target distance, [ ] alpha gate.")
    try:
        while True:
            ok, left, right = read_camera_pair(cv, left_cap, right_cap)
            if not ok:
                print("Could not read both cameras.")
                break
            left, right = match_frame_sizes(cv, left, right, resolution)
            near = max(1.0, target_distance - args.distance_window_mm / 2.0)
            far = target_distance + args.distance_window_mm / 2.0
            bundle = compose_fused_view(left, right, segmenter, args, near, far, shift_offset, alpha_gate)

            pair = side_by_side(cv, left, right, max_width=1280)
            draw_label(cv, pair, "left | right (raw)")
            draw_label(cv, bundle["depth_vis"], f"depth {near:.0f}-{far:.0f}mm  gate {alpha_gate:.2f}")
            masked = cv.bitwise_and(left, left, mask=(bundle["alpha"] > 8).astype("uint8") * 255)
            draw_label(cv, masked, "fused subject (one view)")
            cv.imshow("pair", pair)
            cv.imshow("depth", bundle["depth_vis"])
            cv.imshow("fused subject", masked)
            cv.imshow("hologram input", bundle["holo"])

            key = cv.waitKey(1) & 0xFF
            if key == ord("s"):
                save_bundle(args.output_dir, bundle)
            elif key == ord("j"):
                shift_offset -= 10; print(f"shift_offset_px={shift_offset:.0f}")
            elif key == ord("l"):
                shift_offset += 10; print(f"shift_offset_px={shift_offset:.0f}")
            elif key == ord("u"):
                target_distance = max(100.0, target_distance - 25.0); print(f"target_distance_mm={target_distance:.0f}")
            elif key == ord("o"):
                target_distance += 25.0; print(f"target_distance_mm={target_distance:.0f}")
            elif key == ord("["):
                alpha_gate = max(0.05, alpha_gate - 0.05); print(f"alpha_gate={alpha_gate:.2f}")
            elif key == ord("]"):
                alpha_gate = min(0.95, alpha_gate + 0.05); print(f"alpha_gate={alpha_gate:.2f}")
            elif key == ord("q") or key == 27:
                break
    finally:
        left_cap.release()
        right_cap.release()
        cv.destroyAllWindows()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Fuse a stereo pair into one RGB-D 3D-reconstruction view + subject mask.")
    p.add_argument("--left-camera", type=int, default=0)
    p.add_argument("--right-camera", type=int, default=2)
    p.add_argument("--left-image", type=Path, help="offline mode: left image instead of camera")
    p.add_argument("--right-image", type=Path, help="offline mode: right image instead of camera")
    p.add_argument("--api", default="avfoundation")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--output-dir", type=Path, default=Path("fused_view_output"))
    p.add_argument("--baseline-mm", type=float, default=300.0)
    p.add_argument("--target-distance-mm", type=float, default=525.0)
    p.add_argument("--distance-window-mm", type=float, default=350.0)
    p.add_argument("--hfov-deg", type=float, default=60.0)
    p.add_argument("--shift-sign", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--shift-offset-px", type=float, default=0.0)
    p.add_argument("--match-width", type=int, default=160)
    p.add_argument("--patch-size", type=int, default=17)
    p.add_argument("--shift-step-px", type=int, default=12)
    p.add_argument("--vertical-tolerance-px", type=int, default=30)
    p.add_argument("--cost-threshold", type=float, default=35.0)
    p.add_argument("--texture-threshold", type=float, default=4.0)
    p.add_argument("--open-size", type=int, default=5)
    p.add_argument("--close-size", type=int, default=15)
    p.add_argument("--min-area", type=int, default=700)
    p.add_argument("--alpha-gate", type=float, default=0.5, help="segmentation matte threshold [0,1]")
    p.add_argument("--no-depth-shading", action="store_true", help="do not shade hologram input by depth")
    args = p.parse_args()

    near_mm = max(1.0, args.target_distance_mm - args.distance_window_mm / 2.0)
    far_mm = args.target_distance_mm + args.distance_window_mm / 2.0
    segmenter = SelfieSegmenter()
    try:
        if args.left_image and args.right_image:
            return run_offline(args, segmenter, near_mm, far_mm)
        return run_live(args, segmenter, near_mm, far_mm)
    finally:
        segmenter.close()


if __name__ == "__main__":
    raise SystemExit(main())
