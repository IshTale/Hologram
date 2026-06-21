from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NumPy is required. Install dependencies with:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc
    return np


def require_cv2():
    try:
        import cv2 as cv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenCV is required. Install dependencies with:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc
    return cv


def parse_size(value: str, name: str = "size") -> Tuple[int, int]:
    cleaned = value.lower().replace(",", "x").replace(" ", "")
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise ValueError(f"{name} must look like WIDTHxHEIGHT, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive, got {value!r}")
    return width, height


def parse_optional_resolution(width: Optional[int], height: Optional[int]) -> Optional[Tuple[int, int]]:
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("pass both --width and --height, or neither")
    return int(width), int(height)


def api_preference(cv, api_name: str) -> int:
    normalized = api_name.strip().lower()
    if normalized in {"any", "auto", ""}:
        return cv.CAP_ANY
    api_map = {
        "avfoundation": "CAP_AVFOUNDATION",
        "qt": "CAP_QT",
        "dshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
    }
    attr = api_map.get(normalized)
    if not attr or not hasattr(cv, attr):
        names = ", ".join(sorted(api_map))
        raise ValueError(f"unknown or unsupported OpenCV capture API {api_name!r}; try one of: any, {names}")
    return getattr(cv, attr)


def open_camera(
    cv,
    index: int,
    api: int,
    resolution: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
    warmup_frames: int = 3,
):
    cap = cv.VideoCapture(index, api)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {index}")

    if resolution:
        width, height = resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv.CAP_PROP_FPS, fps)

    for _ in range(max(0, warmup_frames)):
        cap.read()
        time.sleep(0.02)
    return cap


def read_camera_pair(cv, left_cap, right_cap):
    left_ok = left_cap.grab()
    right_ok = right_cap.grab()
    if not left_ok or not right_ok:
        return False, None, None

    left_ok, left = left_cap.retrieve()
    right_ok, right = right_cap.retrieve()
    if not left_ok or not right_ok or left is None or right is None:
        return False, None, None
    return True, left, right


def resize_to(cv, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = size
    if (width, height) == (target_width, target_height):
        return image
    return cv.resize(image, (target_width, target_height), interpolation=cv.INTER_AREA)


def match_frame_sizes(cv, left: np.ndarray, right: np.ndarray, target_size: Optional[Tuple[int, int]] = None):
    if target_size is None:
        target_size = (left.shape[1], left.shape[0])
    return resize_to(cv, left, target_size), resize_to(cv, right, target_size)


def draw_label(cv, image: np.ndarray, text: str, origin=(12, 28), color=(255, 255, 255)):
    cv.putText(image, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv.LINE_AA)


def side_by_side(cv, left: np.ndarray, right: np.ndarray, max_width: int = 1600) -> np.ndarray:
    np = require_numpy()
    left, right = match_frame_sizes(cv, left, right)
    combined = np.hstack([left, right])
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        combined = cv.resize(combined, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    return combined


def find_chessboard(cv, image: np.ndarray, board_size: Tuple[int, int], refine: bool = True):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv.findChessboardCorners(gray, board_size, flags)
    if found and refine:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def make_object_points(board_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    np = require_numpy()
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp


def calibration_image_size(calibration: Dict[str, np.ndarray]) -> Tuple[int, int]:
    raw = calibration["image_size"]
    return int(raw[0]), int(raw[1])


def load_calibration(path: Path) -> Dict[str, np.ndarray]:
    np = require_numpy()
    data = np.load(str(path))
    return {key: data[key] for key in data.files}


def build_rectification_maps(cv, calibration: Dict[str, np.ndarray]):
    image_size = calibration_image_size(calibration)
    left_maps = cv.initUndistortRectifyMap(
        calibration["M1"],
        calibration["D1"],
        calibration["R1"],
        calibration["P1"],
        image_size,
        cv.CV_16SC2,
    )
    right_maps = cv.initUndistortRectifyMap(
        calibration["M2"],
        calibration["D2"],
        calibration["R2"],
        calibration["P2"],
        image_size,
        cv.CV_16SC2,
    )
    return left_maps, right_maps


def rectify_pair(cv, left: np.ndarray, right: np.ndarray, left_maps, right_maps):
    rect_left = cv.remap(left, left_maps[0], left_maps[1], cv.INTER_LINEAR)
    rect_right = cv.remap(right, right_maps[0], right_maps[1], cv.INTER_LINEAR)
    return rect_left, rect_right


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def ensure_num_disparities(value: int) -> int:
    value = max(16, int(value))
    return int(math.ceil(value / 16) * 16)


def calibration_baseline_mm(calibration: Dict[str, np.ndarray]) -> float:
    np = require_numpy()
    return float(np.linalg.norm(calibration["T"]))


def calibration_focal_px(calibration: Dict[str, np.ndarray]) -> float:
    if "P1" in calibration:
        return float(calibration["P1"][0, 0])
    return float(calibration["M1"][0, 0])


def recommend_disparity_range(
    calibration: Dict[str, np.ndarray],
    target_distance_mm: float,
    distance_window_mm: float,
    margin_fraction: float = 0.25,
):
    focal_px = calibration_focal_px(calibration)
    baseline_mm = calibration_baseline_mm(calibration)
    image_width = calibration_image_size(calibration)[0]
    target_distance_mm = max(1.0, float(target_distance_mm))
    half_window = max(1.0, float(distance_window_mm) / 2.0)
    near_mm = max(1.0, target_distance_mm - half_window)
    far_mm = target_distance_mm + half_window

    disparity_near = focal_px * baseline_mm / near_mm
    disparity_far = focal_px * baseline_mm / far_mm
    low_unclipped = max(0, int(math.floor(disparity_far * (1.0 - margin_fraction))))
    high_unclipped = int(math.ceil(disparity_near * (1.0 + margin_fraction)))
    low = min(low_unclipped, max(0, image_width - 17))
    high = min(high_unclipped, image_width - 1)
    if high <= low:
        low = max(0, min(low, image_width - 17))
        high = min(image_width - 1, low + 16)

    num_disparities = ensure_num_disparities(high - low)
    if low + num_disparities > image_width:
        num_disparities = max(16, ((image_width - low) // 16) * 16)
        if num_disparities < 16:
            low = max(0, image_width - 16)
            num_disparities = 16

    return {
        "baseline_mm": baseline_mm,
        "focal_px": focal_px,
        "image_width": image_width,
        "near_mm": near_mm,
        "far_mm": far_mm,
        "expected_disparity_near": disparity_near,
        "expected_disparity_far": disparity_far,
        "min_disparity": low,
        "num_disparities": num_disparities,
        "range_clipped": high_unclipped > image_width or low_unclipped >= image_width,
    }


def create_sgbm(cv, num_disparities: int, block_size: int, min_disparity: int = 0, uniqueness_ratio: int = 10):
    block_size = ensure_odd(block_size)
    num_disparities = ensure_num_disparities(num_disparities)
    channels = 1
    mode = getattr(cv, "STEREO_SGBM_MODE_SGBM_3WAY", cv.STEREO_SGBM_MODE_SGBM)
    return cv.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size * block_size,
        P2=32 * channels * block_size * block_size,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=100,
        speckleRange=2,
        mode=mode,
    )


def compute_disparity(cv, stereo, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    np = require_numpy()
    left_gray = cv.cvtColor(left_bgr, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_bgr, cv.COLOR_BGR2GRAY)
    return stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0


def disparity_preview(cv, disparity: np.ndarray, min_disparity: float, num_disparities: float) -> np.ndarray:
    np = require_numpy()
    scaled = (disparity - float(min_disparity)) / float(num_disparities)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return cv.applyColorMap(scaled, cv.COLORMAP_TURBO)


def point_cloud_from_disparity(cv, calibration: Dict[str, np.ndarray], disparity: np.ndarray) -> np.ndarray:
    return cv.reprojectImageTo3D(disparity, calibration["Q"])


def select_point_cloud(
    points: np.ndarray,
    colors_bgr: np.ndarray,
    disparity: np.ndarray,
    min_disparity: float,
    max_depth_mm: Optional[float],
    stride: int = 2,
    max_points: int = 250000,
):
    np = require_numpy()
    colors_rgb = colors_bgr[:, :, ::-1]
    mask = np.isfinite(points).all(axis=2)
    mask &= np.isfinite(disparity)
    mask &= disparity > float(min_disparity)
    if max_depth_mm is not None and max_depth_mm > 0:
        mask &= np.abs(points[:, :, 2]) <= float(max_depth_mm)

    if stride > 1:
        stride_mask = np.zeros(mask.shape, dtype=bool)
        stride_mask[::stride, ::stride] = True
        mask &= stride_mask

    out_points = points[mask]
    out_colors = colors_rgb[mask]
    if len(out_points) > max_points:
        step = int(math.ceil(len(out_points) / max_points))
        out_points = out_points[::step]
        out_colors = out_colors[::step]
    return out_points, out_colors


def write_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    np = require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.reshape(-1, 3)
    colors_rgb = colors_rgb.reshape(-1, 3).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors_rgb):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def timestamp_name(prefix: str, suffix: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{suffix}"


def print_stderr(lines: Iterable[str]) -> None:
    for line in lines:
        print(line, file=sys.stderr)
