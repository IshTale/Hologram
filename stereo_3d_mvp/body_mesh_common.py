from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from stereo_common import require_cv2, require_numpy


@dataclass
class Mesh:
    name: str
    vertices: object
    faces: List[Tuple[int, int, int]]
    edges: List[Tuple[int, int]]


def require_mediapipe():
    try:
        import mediapipe as mp
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MediaPipe is required for face/hand landmark meshes. Install with:\n"
            "  python -m pip install -r requirements.txt\n\n"
            "If MediaPipe has no wheel for your Python version, create the venv with Python 3.11 or 3.12."
        ) from exc
    return mp


def landmark_vertices(landmarks, image_shape, depth_scale: float = 1.0):
    np = require_numpy()
    height, width = image_shape[:2]
    vertices = []
    for lm in landmarks:
        x = (float(lm.x) - 0.5) * width
        y = -(float(lm.y) - 0.5) * height
        z = -float(lm.z) * width * float(depth_scale)
        vertices.append((x, y, z))
    return np.asarray(vertices, dtype=float)


def sorted_edges(connections: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    edges = set()
    for a, b in connections:
        if a == b:
            continue
        edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def triangles_from_edges(edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    triangles = set()
    for a, b in edges:
        common = adjacency.get(a, set()) & adjacency.get(b, set())
        for c in common:
            tri = tuple(sorted((a, b, c)))
            if len(set(tri)) == 3:
                triangles.add(tri)
    return sorted(triangles)


def hand_surface_faces() -> List[Tuple[int, int, int]]:
    faces = [
        (0, 1, 5),
        (0, 5, 9),
        (0, 9, 13),
        (0, 13, 17),
        (5, 9, 13),
        (5, 13, 17),
        (1, 2, 5),
        (2, 5, 6),
        (2, 3, 6),
        (3, 6, 7),
        (3, 4, 7),
        (4, 7, 8),
    ]
    finger_pairs = [
        ([5, 6, 7, 8], [9, 10, 11, 12]),
        ([9, 10, 11, 12], [13, 14, 15, 16]),
        ([13, 14, 15, 16], [17, 18, 19, 20]),
    ]
    for left, right in finger_pairs:
        for i in range(3):
            faces.append((left[i], right[i], right[i + 1]))
            faces.append((left[i], right[i + 1], left[i + 1]))
    return faces


def write_obj(path: Path, meshes: Sequence[Mesh]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offset = 1
    with path.open("w", encoding="utf-8") as f:
        f.write("# Simple MediaPipe landmark mesh\n")
        for mesh in meshes:
            f.write(f"o {mesh.name}\n")
            for x, y, z in mesh.vertices:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in mesh.faces:
                f.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")
            for a, b in mesh.edges:
                f.write(f"l {a + vertex_offset} {b + vertex_offset}\n")
            vertex_offset += len(mesh.vertices)


def write_ply(path: Path, meshes: Sequence[Mesh]) -> None:
    np = require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)

    all_vertices = []
    all_faces = []
    offset = 0
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=float)
        all_vertices.extend(vertices.tolist())
        for face in mesh.faces:
            all_faces.append(tuple(int(i + offset) for i in face))
        offset += len(vertices)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z in all_vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in all_faces:
            f.write(f"3 {a} {b} {c}\n")


def write_mesh(path: Path, meshes: Sequence[Mesh]) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        write_ply(path, meshes)
        return path
    elif suffix in {"", ".obj"}:
        actual_path = path if suffix else path.with_suffix(".obj")
        write_obj(actual_path, meshes)
        return actual_path
    else:
        raise ValueError("mesh output must end with .obj or .ply")


class LandmarkMeshDetector:
    def __init__(
        self,
        part: str = "auto",
        static_image_mode: bool = True,
        max_faces: int = 1,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_face_landmarks: bool = False,
    ):
        if part not in {"auto", "face", "hand"}:
            raise ValueError("--part must be auto, face, or hand")

        self.cv = require_cv2()
        self.mp = require_mediapipe()
        self.part = part
        self.face_mesh = None
        self.hands = None

        if part in {"auto", "face"}:
            self.face_mesh = self.mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_faces,
                refine_landmarks=refine_face_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.face_edges = sorted_edges(self.mp.solutions.face_mesh.FACEMESH_TESSELATION)
            self.face_faces = triangles_from_edges(self.face_edges)

        if part in {"auto", "hand"}:
            self.hands = self.mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.hand_edges = sorted_edges(self.mp.solutions.hands.HAND_CONNECTIONS)
            self.hand_faces = hand_surface_faces()

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.hands is not None:
            self.hands.close()

    def process(self, image_bgr, depth_scale: float = 1.0):
        cv = self.cv
        preview = image_bgr.copy()
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        meshes: List[Mesh] = []

        if self.face_mesh is not None:
            result = self.face_mesh.process(image_rgb)
            if result.multi_face_landmarks:
                for index, face_landmarks in enumerate(result.multi_face_landmarks):
                    vertices = landmark_vertices(face_landmarks.landmark, image_bgr.shape, depth_scale)
                    meshes.append(Mesh(f"face_{index}", vertices, self.face_faces, self.face_edges))
                    self.mp.solutions.drawing_utils.draw_landmarks(
                        image=preview,
                        landmark_list=face_landmarks,
                        connections=self.mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

        if self.hands is not None:
            result = self.hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    vertices = landmark_vertices(hand_landmarks.landmark, image_bgr.shape, depth_scale)
                    meshes.append(Mesh(f"hand_{index}", vertices, self.hand_faces, self.hand_edges))
                    self.mp.solutions.drawing_utils.draw_landmarks(
                        preview,
                        hand_landmarks,
                        self.mp.solutions.hands.HAND_CONNECTIONS,
                    )

        return meshes, preview


def default_preview_path(output_path: Path) -> Path:
    stem = output_path.stem if output_path.suffix else output_path.name
    parent = output_path.parent if output_path.parent != Path("") else Path(".")
    return parent / f"{stem}_preview.png"


def describe_meshes(meshes: Sequence[Mesh]) -> str:
    parts = []
    for mesh in meshes:
        parts.append(f"{mesh.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return "; ".join(parts)
