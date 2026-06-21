from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

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


def focal_from_hfov(width_px: int, hfov_deg: float) -> float:
    hfov_rad = math.radians(float(hfov_deg))
    return float(width_px) / (2.0 * math.tan(hfov_rad / 2.0))


def shift_range_px(width_px: int, baseline_mm: float, near_mm: float, far_mm: float, hfov_deg: float) -> Tuple[float, float, float]:
    focal_px = focal_from_hfov(width_px, hfov_deg)
    near_mm = max(1.0, float(near_mm))
    far_mm = max(near_mm + 1.0, float(far_mm))
    shift_near = focal_px * float(baseline_mm) / near_mm
    shift_far = focal_px * float(baseline_mm) / far_mm
    return focal_px, min(shift_far, shift_near), max(shift_far, shift_near)


def local_texture(gray, patch_size: int):
    cv = require_cv2()
    np = require_numpy()
    gray_f = gray.astype("float32")
    mean = cv.boxFilter(gray_f, -1, (patch_size, patch_size), normalize=True)
    mean_sq = cv.boxFilter(gray_f * gray_f, -1, (patch_size, patch_size), normalize=True)
    variance = np.maximum(mean_sq - mean * mean, 0)
    return np.sqrt(variance)


def shifted_match_cost(left_gray, right_gray, dx: int, dy: int, patch_size: int):
    cv = require_cv2()
    np = require_numpy()
    h, w = left_gray.shape[:2]
    cost = np.full((h, w), 255.0, dtype="float32")

    lx0 = max(0, -dx)
    lx1 = min(w, w - dx)
    rx0 = lx0 + dx
    rx1 = lx1 + dx

    ly0 = max(0, -dy)
    ly1 = min(h, h - dy)
    ry0 = ly0 + dy
    ry1 = ly1 + dy

    if lx1 <= lx0 or ly1 <= ly0:
        return cost

    left_roi = left_gray[ly0:ly1, lx0:lx1].astype("float32")
    right_roi = right_gray[ry0:ry1, rx0:rx1].astype("float32")
    diff = cv.absdiff(left_roi, right_roi)
    sad = cv.boxFilter(diff, -1, (patch_size, patch_size), normalize=True)
    cost[ly0:ly1, lx0:lx1] = sad
    return cost


def clean_mask(mask, open_size: int, close_size: int, min_area: int):
    cv = require_cv2()
    np = require_numpy()
    out = mask.astype("uint8")
    if open_size > 1:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_size, open_size))
        out = cv.morphologyEx(out, cv.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_size, close_size))
        out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel)
    if min_area > 0:
        count, labels, stats, _ = cv.connectedComponentsWithStats(out, 8)
        keep = np.zeros_like(out)
        for label in range(1, count):
            if stats[label, cv.CC_STAT_AREA] >= min_area:
                keep[labels == label] = 255
        out = keep
    return out


def compute_raw_depth_mask(
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
    open_size: int,
    close_size: int,
    min_area: int,
):
    cv = require_cv2()
    np = require_numpy()
    h, w = left_bgr.shape[:2]
    focal_px, shift_min, shift_max = shift_range_px(w, baseline_mm, near_mm, far_mm, hfov_deg)

    scale = min(1.0, float(match_width) / float(w))
    small_size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
    left_small = cv.resize(left_bgr, small_size, interpolation=cv.INTER_AREA)
    right_small = cv.resize(right_bgr, small_size, interpolation=cv.INTER_AREA)
    left_gray = cv.cvtColor(left_small, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_small, cv.COLOR_BGR2GRAY)
    left_gray = cv.GaussianBlur(left_gray, (3, 3), 0)
    right_gray = cv.GaussianBlur(right_gray, (3, 3), 0)

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
    best_shift = np.zeros(left_gray.shape, dtype="float32")
    for dx in signed_shifts:
        for dy in vertical_shifts:
            cost = shifted_match_cost(left_gray, right_gray, dx, dy, patch_s)
            better = cost < min_cost
            min_cost[better] = cost[better]
            best_shift[better] = abs(float(dx)) / max(scale, 1e-6)

    texture = local_texture(left_gray, patch_s)
    mask_small = ((min_cost <= float(cost_threshold)) & (texture >= float(texture_threshold))).astype("uint8") * 255
    min_area_s = int(round(float(min_area) * scale * scale))
    mask_small = clean_mask(mask_small, open_size=3 if open_size > 0 else 0, close_size=5 if close_size > 0 else 0, min_area=min_area_s)
    mask = cv.resize(mask_small, (w, h), interpolation=cv.INTER_NEAREST)
    mask = clean_mask(mask, open_size=open_size, close_size=close_size, min_area=min_area)

    cost_preview = cv.normalize(min_cost, None, 255, 0, cv.NORM_MINMAX).astype("uint8")
    cost_preview = cv.resize(cost_preview, (w, h), interpolation=cv.INTER_NEAREST)
    cost_preview = cv.applyColorMap(cost_preview, cv.COLORMAP_TURBO)

    best_shift_preview = cv.normalize(best_shift, None, 0, 255, cv.NORM_MINMAX).astype("uint8")
    best_shift_preview = cv.resize(best_shift_preview, (w, h), interpolation=cv.INTER_NEAREST)
    best_shift_preview = cv.applyColorMap(best_shift_preview, cv.COLORMAP_VIRIDIS)

    stats = {
        "focal_px": focal_px,
        "shift_min": shift_min,
        "shift_max": shift_max,
        "scale": scale,
        "candidate_count": len(signed_shifts) * len(list(vertical_shifts)),
        "mask_pixels": int((mask > 0).sum()),
        "shift_min_s": min(signed_shifts),
        "shift_max_s": max(signed_shifts),
    }
    return mask, cost_preview, best_shift_preview, stats


def alpha_cutout(left_bgr, mask):
    cv = require_cv2()
    rgba = cv.cvtColor(left_bgr, cv.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw two-camera depth-ish mask without stereo rectification.")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=2)
    parser.add_argument("--api", default="avfoundation", help="OpenCV capture API: avfoundation or any")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("raw_depth_mask_output"))
    parser.add_argument("--baseline-mm", type=float, default=300.0)
    parser.add_argument("--target-distance-mm", type=float, default=525.0)
    parser.add_argument("--distance-window-mm", type=float, default=350.0)
    parser.add_argument("--hfov-deg", type=float, default=60.0, help="horizontal camera FOV estimate used without calibration")
    parser.add_argument("--shift-sign", type=int, choices=[-1, 1], default=-1, help="-1 is typical when camera 0 is left")
    parser.add_argument("--shift-offset-px", type=float, default=0.0, help="manual x-shift correction in original pixels")
    parser.add_argument("--match-width", type=int, default=160, help="downsample width for raw block search")
    parser.add_argument("--patch-size", type=int, default=17, help="patch size in original pixels")
    parser.add_argument("--shift-step-px", type=int, default=16)
    parser.add_argument("--vertical-tolerance-px", type=int, default=30)
    parser.add_argument("--cost-threshold", type=float, default=35.0)
    parser.add_argument("--texture-threshold", type=float, default=4.0)
    parser.add_argument("--open-size", type=int, default=5)
    parser.add_argument("--close-size", type=int, default=15)
    parser.add_argument("--min-area", type=int, default=700)
    args = parser.parse_args()

    cv = require_cv2()
    api = api_preference(cv, args.api)
    resolution = parse_optional_resolution(args.width, args.height)
    near_mm = max(1.0, args.target_distance_mm - args.distance_window_mm / 2.0)
    far_mm = args.target_distance_mm + args.distance_window_mm / 2.0

    left_cap = open_camera(cv, args.left_camera, api, resolution, args.fps)
    right_cap = open_camera(cv, args.right_camera, api, resolution, args.fps)

    print("Controls: s save mask/cutout, q quit.")
    print("Tune live: j/l shift left/right, i/k vertical tolerance down/up, u/o target distance near/far.")
    print("If the mask is empty, try --shift-sign 1 or move the cameras/person so both cameras see the subject.")

    shift_offset = float(args.shift_offset_px)
    vertical_tolerance = int(args.vertical_tolerance_px)
    target_distance = float(args.target_distance_mm)

    try:
        while True:
            ok, left, right = read_camera_pair(cv, left_cap, right_cap)
            if not ok:
                print("Could not read both cameras.")
                break

            left, right = match_frame_sizes(cv, left, right, resolution)
            near_mm = max(1.0, target_distance - args.distance_window_mm / 2.0)
            far_mm = target_distance + args.distance_window_mm / 2.0
            mask, cost_preview, shift_preview, stats = compute_raw_depth_mask(
                left,
                right,
                baseline_mm=args.baseline_mm,
                near_mm=near_mm,
                far_mm=far_mm,
                hfov_deg=args.hfov_deg,
                shift_sign=args.shift_sign,
                shift_offset_px=shift_offset,
                match_width=args.match_width,
                patch_size=args.patch_size,
                shift_step_px=args.shift_step_px,
                vertical_tolerance_px=vertical_tolerance,
                cost_threshold=args.cost_threshold,
                texture_threshold=args.texture_threshold,
                open_size=args.open_size,
                close_size=args.close_size,
                min_area=args.min_area,
            )
            masked = cv.bitwise_and(left, left, mask=mask)
            mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

            pair_preview = side_by_side(cv, left, right, max_width=1280)
            draw_label(cv, pair_preview, "raw left | raw right")
            draw_label(
                cv,
                mask_bgr,
                f"mask {near_mm:.0f}-{far_mm:.0f}mm shift {stats['shift_min']:.0f}-{stats['shift_max']:.0f}px offset {shift_offset:.0f}px",
            )
            draw_label(cv, masked, "masked left feed")
            draw_label(cv, cost_preview, f"match cost, candidates {stats['candidate_count']}")
            draw_label(cv, shift_preview, "best raw shift")

            cv.imshow("raw cameras", pair_preview)
            cv.imshow("raw depth mask", mask_bgr)
            cv.imshow("masked left", masked)
            cv.imshow("match cost", cost_preview)
            cv.imshow("best shift", shift_preview)

            key = cv.waitKey(1) & 0xFF
            if key == ord("s"):
                args.output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = args.output_dir / timestamp_name("mask", ".png")
                masked_path = args.output_dir / timestamp_name("masked_left", ".png")
                cutout_path = args.output_dir / timestamp_name("cutout_alpha", ".png")
                cv.imwrite(str(mask_path), mask)
                cv.imwrite(str(masked_path), masked)
                cv.imwrite(str(cutout_path), alpha_cutout(left, mask))
                print(f"saved {mask_path}, {masked_path}, {cutout_path}")
            elif key == ord("j"):
                shift_offset -= 10
                print(f"shift_offset_px={shift_offset:.0f}")
            elif key == ord("l"):
                shift_offset += 10
                print(f"shift_offset_px={shift_offset:.0f}")
            elif key == ord("i"):
                vertical_tolerance += 5
                print(f"vertical_tolerance_px={vertical_tolerance}")
            elif key == ord("k"):
                vertical_tolerance = max(0, vertical_tolerance - 5)
                print(f"vertical_tolerance_px={vertical_tolerance}")
            elif key == ord("u"):
                target_distance = max(100.0, target_distance - 25.0)
                print(f"target_distance_mm={target_distance:.0f}")
            elif key == ord("o"):
                target_distance += 25.0
                print(f"target_distance_mm={target_distance:.0f}")
            elif key == ord("q") or key == 27:
                break
    finally:
        left_cap.release()
        right_cap.release()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
