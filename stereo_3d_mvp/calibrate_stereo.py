from __future__ import annotations

import argparse
from pathlib import Path

from stereo_common import find_chessboard, make_object_points, parse_size, require_cv2, require_numpy


def pair_files(pairs_dir: Path):
    left_dir = pairs_dir / "left"
    right_dir = pairs_dir / "right"
    pairs = []
    for left in sorted(left_dir.glob("left_*.png")):
        suffix = left.name.replace("left_", "")
        right = right_dir / f"right_{suffix}"
        if right.exists():
            pairs.append((left, right))
    return pairs


def draw_rectified_preview(cv, left, right, maps_left, maps_right, output_path: Path) -> None:
    np = require_numpy()
    rect_left = cv.remap(left, maps_left[0], maps_left[1], cv.INTER_LINEAR)
    rect_right = cv.remap(right, maps_right[0], maps_right[1], cv.INTER_LINEAR)
    canvas = np.hstack([rect_left, rect_right])
    for y in range(0, canvas.shape[0], 32):
        cv.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a stereo camera pair from chessboard captures.")
    parser.add_argument("--pairs", type=Path, default=Path("calibration_pairs"))
    parser.add_argument("--board-size", default="9x6", help="internal chessboard corners, e.g. 9x6")
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=Path("stereo_calibration.npz"))
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--preview", type=Path, default=Path("rectified_preview.png"))
    parser.add_argument(
        "--rectify-alpha",
        type=float,
        default=1.0,
        help="0 crops to the valid overlap; 1 keeps the full rectified view with black borders",
    )
    args = parser.parse_args()

    cv = require_cv2()
    np = require_numpy()
    board_size = parse_size(args.board_size, "board size")
    object_template = make_object_points(board_size, args.square_size_mm)

    object_points = []
    left_points = []
    right_points = []
    image_size = None
    first_good_images = None

    pairs = pair_files(args.pairs)
    if not pairs:
        raise SystemExit(f"no pairs found under {args.pairs}/left and {args.pairs}/right")

    print(f"found {len(pairs)} candidate pairs")
    for left_path, right_path in pairs:
        left = cv.imread(str(left_path))
        right = cv.imread(str(right_path))
        if left is None or right is None:
            print(f"skip unreadable pair: {left_path.name}, {right_path.name}")
            continue
        if left.shape[:2] != right.shape[:2]:
            print(f"skip mismatched size pair: {left_path.name}, {right_path.name}")
            continue

        pair_size = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = pair_size
        elif image_size != pair_size:
            print(f"skip size {pair_size}; expected {image_size}: {left_path.name}")
            continue

        found_left, corners_left = find_chessboard(cv, left, board_size, refine=True)
        found_right, corners_right = find_chessboard(cv, right, board_size, refine=True)
        if not found_left or not found_right:
            print(f"skip board miss: {left_path.name}, {right_path.name}")
            continue

        object_points.append(object_template.copy())
        left_points.append(corners_left)
        right_points.append(corners_right)
        if first_good_images is None:
            first_good_images = (left, right)
        print(f"accepted {left_path.name}")

    if image_size is None:
        raise SystemExit("no readable, same-size image pairs were found")
    if len(object_points) < args.min_pairs:
        raise SystemExit(
            f"only {len(object_points)} usable pairs; capture at least {args.min_pairs} with the board visible in both views"
        )

    print(f"calibrating with {len(object_points)} usable pairs at {image_size[0]}x{image_size[1]}")
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    rms_left, m1, d1, _, _ = cv.calibrateCamera(object_points, left_points, image_size, None, None)
    rms_right, m2, d2, _, _ = cv.calibrateCamera(object_points, right_points, image_size, None, None)

    stereo_flags = cv.CALIB_FIX_INTRINSIC
    rms_stereo, m1, d1, m2, d2, r, t, e, f = cv.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        m1,
        d1,
        m2,
        d2,
        image_size,
        criteria=criteria,
        flags=stereo_flags,
    )

    r1, r2, p1, p2, q, roi1, roi2 = cv.stereoRectify(
        m1,
        d1,
        m2,
        d2,
        image_size,
        r,
        t,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=float(args.rectify_alpha),
    )

    np.savez_compressed(
        args.output,
        image_size=np.array(image_size, dtype=np.int32),
        board_size=np.array(board_size, dtype=np.int32),
        square_size_mm=np.array([args.square_size_mm], dtype=np.float32),
        M1=m1,
        D1=d1,
        M2=m2,
        D2=d2,
        R=r,
        T=t,
        E=e,
        F=f,
        R1=r1,
        R2=r2,
        P1=p1,
        P2=p2,
        Q=q,
        roi1=np.array(roi1, dtype=np.int32),
        roi2=np.array(roi2, dtype=np.int32),
        rms_left=np.array([rms_left], dtype=np.float64),
        rms_right=np.array([rms_right], dtype=np.float64),
        rms_stereo=np.array([rms_stereo], dtype=np.float64),
        usable_pairs=np.array([len(object_points)], dtype=np.int32),
        rectify_alpha=np.array([args.rectify_alpha], dtype=np.float32),
    )

    if first_good_images:
        maps_left = cv.initUndistortRectifyMap(m1, d1, r1, p1, image_size, cv.CV_16SC2)
        maps_right = cv.initUndistortRectifyMap(m2, d2, r2, p2, image_size, cv.CV_16SC2)
        draw_rectified_preview(cv, first_good_images[0], first_good_images[1], maps_left, maps_right, args.preview)
        print(f"wrote rectified preview: {args.preview}")

    baseline_mm = float(np.linalg.norm(t))
    print(f"wrote calibration: {args.output}")
    print(f"left RMS={rms_left:.4f}  right RMS={rms_right:.4f}  stereo RMS={rms_stereo:.4f}")
    print(f"estimated baseline={baseline_mm:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
