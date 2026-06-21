from __future__ import annotations

import argparse

from stereo_common import api_preference, require_cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indices.")
    parser.add_argument("--max-index", type=int, default=8, help="highest camera index to try")
    parser.add_argument("--api", default="avfoundation", help="OpenCV capture API: avfoundation or any")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    cv = require_cv2()
    api = api_preference(cv, args.api)

    found = 0
    for index in range(args.max_index + 1):
        cap = cv.VideoCapture(index, api)
        if not cap.isOpened():
            print(f"[{index}] closed")
            continue

        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, frame = cap.read()
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"[{index}] open  backend={backend}  frame={w}x{h}")
            found += 1
        else:
            print(f"[{index}] opened but did not return a frame  backend={backend}")
        cap.release()

    if found == 0:
        print("No readable cameras found. On macOS, allow camera permission for Terminal/Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
