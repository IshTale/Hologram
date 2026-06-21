from __future__ import annotations

import argparse
from pathlib import Path

from body_mesh_common import LandmarkMeshDetector, describe_meshes, write_mesh
from stereo_common import api_preference, draw_label, open_camera, parse_optional_resolution, require_cv2, timestamp_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Live face/hand landmark mesh capture from one camera.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--api", default="avfoundation", help="OpenCV capture API: avfoundation or any")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--part", choices=["auto", "face", "hand"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("body_mesh_output"))
    parser.add_argument("--max-faces", type=int, default=1)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--refine-face-landmarks", action="store_true")
    args = parser.parse_args()

    cv = require_cv2()
    api = api_preference(cv, args.api)
    resolution = parse_optional_resolution(args.width, args.height)
    cap = open_camera(cv, args.camera, api, resolution, args.fps)
    detector = LandmarkMeshDetector(
        part=args.part,
        static_image_mode=False,
        max_faces=args.max_faces,
        max_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        refine_face_landmarks=args.refine_face_landmarks,
    )

    print("Controls: s save OBJ mesh and preview, q quit.")
    print("Tip: use a clear, well-lit face/hand; the exported z is MediaPipe-relative, not metric depth.")

    last_meshes = []
    last_preview = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Could not read camera.")
                break

            meshes, preview = detector.process(frame, depth_scale=args.depth_scale)
            last_meshes = meshes
            last_preview = preview
            label = describe_meshes(meshes) if meshes else "no landmarks"
            draw_label(cv, preview, label[:90], (12, 28))
            draw_label(cv, preview, "s save  q quit", (12, preview.shape[0] - 16))
            cv.imshow("face/hand mesh capture", preview)

            key = cv.waitKey(1) & 0xFF
            if key == ord("s"):
                if not last_meshes:
                    print("not saved: no face/hand landmarks detected")
                    continue
                args.output_dir.mkdir(parents=True, exist_ok=True)
                obj_path = args.output_dir / timestamp_name("body_mesh", ".obj")
                preview_path = args.output_dir / timestamp_name("body_mesh_preview", ".png")
                output_path = write_mesh(obj_path, last_meshes)
                cv.imwrite(str(preview_path), last_preview)
                print(f"saved {output_path}")
            if key == ord("q") or key == 27:
                break
    finally:
        detector.close()
        cap.release()
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
