from __future__ import annotations

import argparse
from pathlib import Path

from stereo_common import (
    api_preference,
    build_rectification_maps,
    compute_disparity,
    create_sgbm,
    disparity_preview,
    draw_label,
    ensure_num_disparities,
    load_calibration,
    match_frame_sizes,
    open_camera,
    parse_optional_resolution,
    point_cloud_from_disparity,
    read_camera_pair,
    rectify_pair,
    require_cv2,
    select_point_cloud,
    side_by_side,
    timestamp_name,
    write_ply,
)


def export_snapshot(cv, output_dir: Path, calibration, rect_left, rect_right, disparity, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = point_cloud_from_disparity(cv, calibration, disparity)
    out_points, out_colors = select_point_cloud(
        points,
        rect_left,
        disparity,
        min_disparity=args.min_disparity,
        max_depth_mm=args.max_depth_mm,
        stride=args.stride,
        max_points=args.max_points,
    )
    ply_path = output_dir / timestamp_name("cloud", ".ply")
    write_ply(ply_path, out_points, out_colors)
    cv.imwrite(str(output_dir / timestamp_name("left_rectified", ".png")), rect_left)
    cv.imwrite(str(output_dir / timestamp_name("right_rectified", ".png")), rect_right)
    print(f"snapshot: wrote {len(out_points)} points to {ply_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live stereo disparity and PLY export from two cameras.")
    parser.add_argument("--calibration", type=Path, default=Path("stereo_calibration.npz"))
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--api", default="avfoundation", help="OpenCV capture API: avfoundation or any")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("recon_output"))
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--max-depth-mm", type=float, default=3000.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=250000)
    args = parser.parse_args()

    cv = require_cv2()
    api = api_preference(cv, args.api)
    resolution = parse_optional_resolution(args.width, args.height)
    calibration = load_calibration(args.calibration)
    calibration_size = tuple(int(x) for x in calibration["image_size"])
    left_maps, right_maps = build_rectification_maps(cv, calibration)
    num_disparities = ensure_num_disparities(args.num_disparities)
    stereo = create_sgbm(
        cv,
        num_disparities=num_disparities,
        block_size=args.block_size,
        min_disparity=args.min_disparity,
        uniqueness_ratio=args.uniqueness_ratio,
    )

    left_cap = open_camera(cv, args.left_camera, api, resolution, args.fps)
    right_cap = open_camera(cv, args.right_camera, api, resolution, args.fps)
    print("Controls: s export PLY snapshot, q quit.")
    print("If the disparity is mostly blank/noisy, swap left/right camera indices or improve lighting/texture.")

    try:
        while True:
            ok, left, right = read_camera_pair(cv, left_cap, right_cap)
            if not ok:
                print("Could not read both cameras.")
                break

            left, right = match_frame_sizes(cv, left, right, calibration_size)
            rect_left, rect_right = rectify_pair(cv, left, right, left_maps, right_maps)
            disparity = compute_disparity(cv, stereo, rect_left, rect_right)
            disp_preview = disparity_preview(cv, disparity, args.min_disparity, num_disparities)

            pair_preview = side_by_side(cv, rect_left, rect_right, max_width=1600)
            draw_label(cv, pair_preview, "rectified left | rectified right")
            draw_label(cv, disp_preview, "disparity")
            cv.imshow("rectified cameras", pair_preview)
            cv.imshow("disparity", disp_preview)

            key = cv.waitKey(1) & 0xFF
            if key == ord("s"):
                export_snapshot(cv, args.output_dir, calibration, rect_left, rect_right, disparity, args)
            if key == ord("q") or key == 27:
                break
    finally:
        left_cap.release()
        right_cap.release()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
