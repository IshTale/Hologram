from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from stereo_common import require_cv2, require_numpy


@dataclass
class StereoMesh:
    vertices: object
    colors_rgb: object
    faces: object

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.faces))


def _edge_lengths(np, a, b):
    return np.linalg.norm(a - b, axis=2)


def depth_map_to_mesh(
    points,
    colors_bgr,
    disparity,
    min_disparity: float = 0.0,
    max_depth_mm: Optional[float] = 2000.0,
    mesh_step: int = 4,
    max_edge_mm: float = 80.0,
    max_faces: int = 200000,
) -> StereoMesh:
    """Convert a rectified stereo depth map into a downsampled triangle mesh."""
    np = require_numpy()
    mesh_step = max(1, int(mesh_step))

    points_s = points[::mesh_step, ::mesh_step]
    colors_s = colors_bgr[::mesh_step, ::mesh_step, ::-1]
    disparity_s = disparity[::mesh_step, ::mesh_step]

    valid = np.isfinite(points_s).all(axis=2)
    valid &= np.isfinite(disparity_s)
    valid &= disparity_s > float(min_disparity)
    if max_depth_mm is not None and max_depth_mm > 0:
        valid &= np.abs(points_s[:, :, 2]) <= float(max_depth_mm)

    indices = np.full(valid.shape, -1, dtype=np.int32)
    indices[valid] = np.arange(int(valid.sum()), dtype=np.int32)

    vertices = points_s[valid].astype(float)
    colors_rgb = colors_s[valid].astype("uint8")

    if valid.shape[0] < 2 or valid.shape[1] < 2 or len(vertices) == 0:
        return StereoMesh(vertices, colors_rgb, np.zeros((0, 3), dtype=np.int32))

    v00 = valid[:-1, :-1]
    v10 = valid[:-1, 1:]
    v01 = valid[1:, :-1]
    v11 = valid[1:, 1:]

    p00 = points_s[:-1, :-1]
    p10 = points_s[:-1, 1:]
    p01 = points_s[1:, :-1]
    p11 = points_s[1:, 1:]

    tri1_ok = v00 & v01 & v10
    tri2_ok = v10 & v01 & v11

    if max_edge_mm and max_edge_mm > 0:
        max_edge = float(max_edge_mm)
        e00_10 = _edge_lengths(np, p00, p10) <= max_edge
        e00_01 = _edge_lengths(np, p00, p01) <= max_edge
        e10_01 = _edge_lengths(np, p10, p01) <= max_edge
        e10_11 = _edge_lengths(np, p10, p11) <= max_edge
        e01_11 = _edge_lengths(np, p01, p11) <= max_edge
        tri1_ok &= e00_10 & e00_01 & e10_01
        tri2_ok &= e10_01 & e10_11 & e01_11

    i00 = indices[:-1, :-1]
    i10 = indices[:-1, 1:]
    i01 = indices[1:, :-1]
    i11 = indices[1:, 1:]

    tri1 = np.stack([i00[tri1_ok], i01[tri1_ok], i10[tri1_ok]], axis=1) if tri1_ok.any() else np.zeros((0, 3), dtype=np.int32)
    tri2 = np.stack([i10[tri2_ok], i01[tri2_ok], i11[tri2_ok]], axis=1) if tri2_ok.any() else np.zeros((0, 3), dtype=np.int32)
    faces = np.vstack([tri1, tri2]).astype(np.int32)

    if max_faces and len(faces) > max_faces:
        keep_step = int(np.ceil(len(faces) / int(max_faces)))
        faces = faces[::keep_step]

    return StereoMesh(vertices, colors_rgb, faces)


def write_obj_mesh(path: Path, mesh: StereoMesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stereo depth mesh generated from two live cameras\n")
        f.write("o stereo_mesh\n")
        for point, color in zip(mesh.vertices, mesh.colors_rgb):
            r, g, b = [int(v) / 255.0 for v in color]
            f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for a, b, c in mesh.faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def write_ply_mesh(path: Path, mesh: StereoMesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {mesh.vertex_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {mesh.face_count}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for point, color in zip(mesh.vertices, mesh.colors_rgb):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for a, b, c in mesh.faces:
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


def write_stereo_mesh(path: Path, mesh: StereoMesh) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        write_ply_mesh(path, mesh)
        return path
    if suffix in {"", ".obj"}:
        actual_path = path if suffix else path.with_suffix(".obj")
        write_obj_mesh(actual_path, mesh)
        return actual_path
    raise ValueError("mesh output must end with .obj or .ply")


def render_mesh_preview(mesh: StereoMesh, width: int = 640, height: int = 480, max_draw_faces: int = 6000):
    cv = require_cv2()
    np = require_numpy()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if mesh.vertex_count == 0:
        return canvas

    vertices = np.asarray(mesh.vertices, dtype=float)
    finite = np.isfinite(vertices).all(axis=1)
    if not finite.any():
        return canvas

    xy = vertices[:, :2].copy()
    mins = np.nanpercentile(xy[finite], 2, axis=0)
    maxs = np.nanpercentile(xy[finite], 98, axis=0)
    span = np.maximum(maxs - mins, 1.0)
    xy = (xy - mins) / span
    xy[:, 0] = xy[:, 0] * (width - 1)
    xy[:, 1] = (1.0 - xy[:, 1]) * (height - 1)
    xy = np.clip(xy, [0, 0], [width - 1, height - 1]).astype(np.int32)

    faces = mesh.faces
    if len(faces) > max_draw_faces:
        step = int(np.ceil(len(faces) / max_draw_faces))
        faces = faces[::step]

    for a, b, c in faces:
        pa = tuple(int(v) for v in xy[int(a)])
        pb = tuple(int(v) for v in xy[int(b)])
        pc = tuple(int(v) for v in xy[int(c)])
        color = (120, 210, 255)
        cv.line(canvas, pa, pb, color, 1, cv.LINE_AA)
        cv.line(canvas, pb, pc, color, 1, cv.LINE_AA)
        cv.line(canvas, pc, pa, color, 1, cv.LINE_AA)

    return canvas


def mesh_summary(mesh: StereoMesh) -> str:
    return f"mesh: {mesh.vertex_count} vertices, {mesh.face_count} faces"
