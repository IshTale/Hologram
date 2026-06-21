from __future__ import annotations

import argparse
from pathlib import Path

from stereo_common import (
    build_rectification_maps,
    compute_disparity,
    create_sgbm,
    disparity_preview,
    ensure_num_disparities,
    load_calibration,
    point_cloud_from_disparity,
    rectify_pair,
    resize_to,
    require_cv2,
    select_point_cloud,
    write_ply,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct a PLY point cloud from one stereo image pair.")
    parser.add_argument("--calibration", type=Path, default=Path("stereo_calibration.npz"))
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("point_cloud.ply"))
    parser.add_argument("--disparity-preview", type=Path, default=Path("disparity_preview.png"))
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--max-depth-mm", type=float, default=3000.0)
    parser.add_argument("--stride", type=int, default=2, help="sample every Nth pixel for the PLY")
    parser.add_argument("--max-points", type=int, default=250000)
    args = parser.parse_args()

    cv = require_cv2()
    calibration = load_calibration(args.calibration)
    image_size = tuple(int(x) for x in calibration["image_size"])
    left_maps, right_maps = build_rectification_maps(cv, calibration)

    left = cv.imread(str(args.left))
    right = cv.imread(str(args.right))
    if left is None or right is None:
        raise SystemExit("could not read one or both input images")
    left = resize_to(cv, left, image_size)
    right = resize_to(cv, right, image_size)

    rect_left, rect_right = rectify_pair(cv, left, right, left_maps, right_maps)
    num_disparities = ensure_num_disparities(args.num_disparities)
    stereo = create_sgbm(
        cv,
        num_disparities=num_disparities,
        block_size=args.block_size,
        min_disparity=args.min_disparity,
        uniqueness_ratio=args.uniqueness_ratio,
    )
    disparity = compute_disparity(cv, stereo, rect_left, rect_right)
    preview = disparity_preview(cv, disparity, args.min_disparity, num_disparities)
    cv.imwrite(str(args.disparity_preview), preview)

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
    write_ply(args.output, out_points, out_colors)
    print(f"wrote {len(out_points)} points to {args.output}")
    print(f"wrote disparity preview to {args.disparity_preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
