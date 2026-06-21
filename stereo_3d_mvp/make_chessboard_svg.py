from __future__ import annotations

import argparse
from pathlib import Path

from stereo_common import parse_size


def make_svg(board_size, square_size_mm: float, margin_mm: float) -> str:
    inner_cols, inner_rows = board_size
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    width = 2 * margin_mm + squares_x * square_size_mm
    height = 2 * margin_mm + squares_y * square_size_mm

    rects = []
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                rx = margin_mm + x * square_size_mm
                ry = margin_mm + y * square_size_mm
                rects.append(
                    f'<rect x="{rx:.3f}mm" y="{ry:.3f}mm" '
                    f'width="{square_size_mm:.3f}mm" height="{square_size_mm:.3f}mm" fill="black"/>'
                )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" height="{height:.3f}mm" '
            f'viewBox="0 0 {width:.3f} {height:.3f}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
            *rects,
            "</svg>",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a printable chessboard SVG for stereo calibration.")
    parser.add_argument("--board-size", default="9x6", help="internal chessboard corners, e.g. 9x6")
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--margin-mm", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("chessboard_9x6_25mm.svg"))
    args = parser.parse_args()

    board_size = parse_size(args.board_size, "board size")
    svg = make_svg(board_size, args.square_size_mm, args.margin_mm)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")
    print("Print at 100% scale. Do not fit-to-page if you need metric depth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
