from __future__ import annotations

import argparse
from pathlib import Path

from stereo_common import (
    api_preference,
    draw_label,
    find_chessboard,
    match_frame_sizes,
    open_camera,
    parse_optional_resolution,
    parse_size,
    read_camera_pair,
    require_cv2,
    side_by_side,
)


def next_pair_index(left_dir: Path) -> int:
    existing = sorted(left_dir.glob("left_*.png"))
    if not existing:
        return 0
    numbers = []
    for path in existing:
        try:
            numbers.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(numbers) + 1) if numbers else 0


def save_pair(cv, left, right, out_dir: Path, index: int) -> None:
    left_dir = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(left_dir / f"left_{index:04d}.png"), left)
    cv.imwrite(str(right_dir / f"right_{index:04d}.png"), right)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture synchronized-ish stereo calibration image pairs.")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--api", default="avfoundation", help="OpenCV capture API: avfoundation or any")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--board-size", default="9x6", help="internal chessboard corners, e.g. 9x6")
    parser.add_argument("--output", type=Path, default=Path("calibration_pairs"))
    parser.add_argument("--count", type=int, default=25, help="stop after this many saved pairs")
    parser.add_argument("--save-anyway", action="store_true", help="allow saving when one side misses the board")
    args = parser.parse_args()

    cv = require_cv2()
    api = api_preference(cv, args.api)
    resolution = parse_optional_resolution(args.width, args.height)
    board_size = parse_size(args.board_size, "board size")

    left_cap = open_camera(cv, args.left_camera, api, resolution, args.fps)
    right_cap = open_camera(cv, args.right_camera, api, resolution, args.fps)
    index = next_pair_index(args.output / "left")
    saved = 0

    print("Controls: SPACE/s save pair, q quit.")
    print("Move the board around: near/far, corners, tilt, and rotate. Both views must see the full board.")

    try:
        while True:
            ok, left, right = read_camera_pair(cv, left_cap, right_cap)
            if not ok:
                print("Could not read both cameras.")
                break

            left, right = match_frame_sizes(cv, left, right, resolution)
            left_found, left_corners = find_chessboard(cv, left, board_size)
            right_found, right_corners = find_chessboard(cv, right, board_size)

            preview_left = left.copy()
            preview_right = right.copy()
            cv.drawChessboardCorners(preview_left, board_size, left_corners, left_found)
            cv.drawChessboardCorners(preview_right, board_size, right_corners, right_found)
            draw_label(cv, preview_left, f"left {args.left_camera}: {'found' if left_found else 'missing'}")
            draw_label(cv, preview_right, f"right {args.right_camera}: {'found' if right_found else 'missing'}")
            preview = side_by_side(cv, preview_left, preview_right)
            draw_label(cv, preview, f"saved {saved}/{args.count}  next index {index:04d}", (12, preview.shape[0] - 16))
            cv.imshow("capture calibration pairs", preview)

            key = cv.waitKey(1) & 0xFF
            should_save = key in (ord(" "), ord("s"))
            if should_save:
                if args.save_anyway or (left_found and right_found):
                    save_pair(cv, left, right, args.output, index)
                    print(f"saved pair {index:04d}")
                    index += 1
                    saved += 1
                    if saved >= args.count:
                        break
                else:
                    print("not saved: chessboard must be detected in both cameras; pass --save-anyway to override")
            if key == ord("q") or key == 27:
                break
    finally:
        left_cap.release()
        right_cap.release()
        cv.destroyAllWindows()

    print(f"done; saved {saved} new pairs under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
