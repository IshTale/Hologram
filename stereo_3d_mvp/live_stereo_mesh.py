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
    recommend_disparity_range,
    read_camera_pair,
    rectify_pair,
    require_cv2,
    side_by_side,
    timestamp_name,
)
from stereo_mesh_common import depth_map_to_mesh, mesh_summary, render_mesh_preview, write_stereo_mesh


def save_mesh_snapshot(cv, output_dir: Path, mesh, rect_left, rect_right, disp_preview, extension: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = extension.lower().lstrip(".")
    mesh_path = output_dir / timestamp_name("stereo_mesh", f".{extension}")
    actual_path = write_stereo_mesh(mesh_path, mesh)
    cv.imwrite(str(output_dir / timestamp_name("left_rectified", ".png")), rect_left)
    cv.imwrite(str(output_dir / timestamp_name("right_rectified", ".png")), rect_right)
    cv.imwrite(str(output_dir / timestamp_name("disparity", ".png")), disp_preview)
    print(f"saved {actual_path} ({mesh.vertex_count} vertices, {mesh.face_count} faces)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time stereo 3D triangle mesh creation from two cameras.")
    parser.add_argument("--calibration", type=Path, default=Path("stereo_calibration.npz"))
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=2)
    parser.add_argument("--api", default="avfoundation", help="OpenCV capture API: avfoundation or any")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("stereo_mesh_output"))
    parser.add_argument("--format", choices=["obj", "ply"], default="obj")
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--target-distance-mm", type=float, help="expected subject distance from the cameras")
    parser.add_argument("--distance-window-mm", type=float, default=300.0, help="depth band around target distance")
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--max-depth-mm", type=float, default=2000.0)
    parser.add_argument("--mesh-step", type=int, default=4, help="larger values are faster and less detailed")
    parser.add_argument("--max-edge-mm", type=float, default=80.0, help="reject triangles with longer 3D edges")
    parser.add_argument("--max-faces", type=int, default=200000)
    parser.add_argument("--save-every", type=int, default=0, help="auto-save every N frames; 0 disables")
    args = parser.parse_args()

    cv = require_cv2()
    if not args.calibration.exists():
        raise SystemExit(
            f"missing calibration file: {args.calibration}\n"
            "Run calibrate_stereo.py first, then rerun live_stereo_mesh.py."
        )

    api = api_preference(cv, args.api)
    resolution = parse_optional_resolution(args.width, args.height)
    calibration = load_calibration(args.calibration)
    calibration_size = tuple(int(x) for x in calibration["image_size"])
    left_maps, right_maps = build_rectification_maps(cv, calibration)
    if args.target_distance_mm:
        recommendation = recommend_disparity_range(
            calibration,
            target_distance_mm=args.target_distance_mm,
            distance_window_mm=args.distance_window_mm,
        )
        args.min_disparity = int(recommendation["min_disparity"])
        args.num_disparities = int(recommendation["num_disparities"])
        print(
            "Geometry-aware disparity: "
            f"baseline={recommendation['baseline_mm']:.1f}mm, "
            f"focal={recommendation['focal_px']:.1f}px, "
            f"image width={recommendation['image_width']}px, "
            f"distance range={recommendation['near_mm']:.0f}-{recommendation['far_mm']:.0f}mm, "
            f"expected disparity={recommendation['expected_disparity_far']:.0f}-"
            f"{recommendation['expected_disparity_near']:.0f}px, "
            f"min={args.min_disparity}, num={args.num_disparities}"
        )
        if recommendation["range_clipped"]:
            print(
                "Warning: the requested distance/baseline implies disparity near or beyond the image width. "
                "If the mesh is empty, move the subject farther away or reduce the camera baseline."
            )
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
    print("Controls: s save current mesh, q quit.")
    print("This creates a live mesh from stereo disparity. Calibrate first and keep cameras rigid.")

    frame_index = 0
    last_mesh = None
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
            points = point_cloud_from_disparity(cv, calibration, disparity)
            mesh = depth_map_to_mesh(
                points,
                rect_left,
                disparity,
                min_disparity=args.min_disparity,
                max_depth_mm=args.max_depth_mm,
                mesh_step=args.mesh_step,
                max_edge_mm=args.max_edge_mm,
                max_faces=args.max_faces,
            )
            last_mesh = mesh

            pair_preview = side_by_side(cv, rect_left, rect_right, max_width=1280)
            mesh_preview = render_mesh_preview(mesh, width=640, height=480)
            draw_label(cv, pair_preview, "rectified left | rectified right")
            draw_label(cv, disp_preview, "disparity")
            draw_label(cv, mesh_preview, mesh_summary(mesh))
            draw_label(cv, mesh_preview, "s save mesh  q quit", (12, mesh_preview.shape[0] - 16))

            cv.imshow("rectified cameras", pair_preview)
            cv.imshow("disparity", disp_preview)
            cv.imshow("live stereo mesh", mesh_preview)

            if args.save_every > 0 and frame_index > 0 and frame_index % args.save_every == 0 and mesh.face_count > 0:
                save_mesh_snapshot(cv, args.output_dir, mesh, rect_left, rect_right, disp_preview, args.format)

            key = cv.waitKey(1) & 0xFF
            if key == ord("s"):
                if last_mesh is None or last_mesh.face_count == 0:
                    print("not saved: mesh is empty; improve calibration, texture, lighting, or disparity settings")
                else:
                    save_mesh_snapshot(cv, args.output_dir, last_mesh, rect_left, rect_right, disp_preview, args.format)
            if key == ord("q") or key == 27:
                break
            frame_index += 1
    finally:
        left_cap.release()
        right_cap.release()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
