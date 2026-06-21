from __future__ import annotations

import argparse
from pathlib import Path

from body_mesh_common import LandmarkMeshDetector, default_preview_path, describe_meshes, write_mesh
from stereo_common import require_cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a simple face or hand 3D landmark mesh from one image.")
    parser.add_argument("--image", type=Path, required=True, help="input image containing a face and/or hand")
    parser.add_argument("--part", choices=["auto", "face", "hand"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("body_mesh.obj"), help="output .obj or .ply mesh")
    parser.add_argument("--preview", type=Path, help="optional landmark overlay image")
    parser.add_argument("--max-faces", type=int, default=1)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--depth-scale", type=float, default=1.0, help="scales MediaPipe relative z depth")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--refine-face-landmarks", action="store_true")
    args = parser.parse_args()

    cv = require_cv2()
    image = cv.imread(str(args.image))
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    detector = LandmarkMeshDetector(
        part=args.part,
        static_image_mode=True,
        max_faces=args.max_faces,
        max_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        refine_face_landmarks=args.refine_face_landmarks,
    )
    try:
        meshes, preview = detector.process(image, depth_scale=args.depth_scale)
    finally:
        detector.close()

    if not meshes:
        raise SystemExit("no face or hand landmarks detected; try a clearer, front-facing, well-lit image")

    output_path = write_mesh(args.output, meshes)
    preview_path = args.preview or default_preview_path(args.output)
    cv.imwrite(str(preview_path), preview)
    print(f"wrote mesh: {output_path}")
    print(f"wrote preview: {preview_path}")
    print(describe_meshes(meshes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
