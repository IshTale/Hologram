# Real-Time Stereo 3D Mesh MVP

This folder centers on the two-camera workflow: a real-time triangle mesh generated from the Mac camera plus a USB camera.

Pipeline:

1. Calibrate the two cameras with a chessboard.
2. Rectify the live camera feeds.
3. Compute stereo disparity.
4. Reproject disparity into 3D points.
5. Connect neighboring valid depth points into triangles.
6. Save the current live mesh as OBJ or PLY.

This matches the multi-view stereo idea from the article you sent: rectification, correspondence, 3D reprojection/triangulation, then mesh creation. For faces or hands, keep the subject still, light it well, and keep the calibrated camera rig rigid.

## Setup

```bash
cd /Users/richik/Desktop/Richik/Hologram/stereo_3d_mvp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core two-camera stereo mesh flow only needs OpenCV and NumPy.

## Real-Time Two-Camera Mesh

You already found the working cameras are `0` and `2`.

Capture calibration pairs:

```bash
python capture_calibration_pairs.py --left-camera 0 --right-camera 2 --board-size 5x8 --width 640 --height 480 --count 25
```

Use SPACE or `s` to save when both views detect the full chessboard. Move the board around: center, corners, tilted, closer, and farther.

For your drawn paper board:

- Squares across: `6`
- Squares down: `9`
- OpenCV internal corners / board size: `5x8`
- Checker size: `1 inch = 25.4 mm`

Calibrate with less aggressive rectification cropping:

```bash
python calibrate_stereo.py --pairs calibration_pairs --board-size 5x8 --square-size-mm 25.4 --rectify-alpha 1.0 --output stereo_calibration.npz
```

`--rectify-alpha 1.0` keeps more of each camera view after rectification. This helps when the face or hand is near the edge of the overlapping view.

Run real-time stereo mesh generation:

```bash
python live_stereo_mesh.py --calibration stereo_calibration.npz --left-camera 0 --right-camera 2 --width 640 --height 480
```

Controls:

- `s` saves the current live mesh into `stereo_mesh_output/`.
- `q` quits.

Output defaults to `.obj`. For `.ply`:

```bash
python live_stereo_mesh.py --calibration stereo_calibration.npz --left-camera 0 --right-camera 2 --format ply
```

Auto-save a mesh every 30 frames:

```bash
python live_stereo_mesh.py --calibration stereo_calibration.npz --left-camera 0 --right-camera 2 --save-every 30
```

Face/hand-focused tuning:

```bash
python live_stereo_mesh.py \
  --calibration stereo_calibration.npz \
  --left-camera 0 \
  --right-camera 2 \
  --width 640 \
  --height 480 \
  --target-distance-mm 525 \
  --distance-window-mm 300 \
  --mesh-step 3 \
  --max-depth-mm 900 \
  --max-edge-mm 45 \
  --format obj
```

With a 300 mm camera baseline and the person about 500-550 mm away, disparity is much larger than the default search range. `--target-distance-mm 525` reads the calibration and automatically chooses `--min-disparity` and `--num-disparities` for that triangle geometry.

Tuning notes:

- Increase `--mesh-step` for faster, lower-detail meshes.
- Decrease `--mesh-step` for denser meshes.
- Lower `--max-depth-mm` to focus on nearby faces/hands and remove background.
- Lower `--max-edge-mm` to reject bad stretched triangles.
- Increase `--num-disparities` if the subject is close and disparity clips.
- If one raw camera cannot see the face/hand at all, move the cameras closer together, toe them inward toward the subject, or move the subject farther away. Stereo can only reconstruct overlap visible in both cameras.

## Raw Two-Camera Depth Mask

Use this when rectification is visually failing. It uses the original camera feeds, not the rectified images. It estimates the expected raw left/right image shift from your baseline and subject distance, searches nearby horizontal and vertical offsets, and produces a rough binary foreground mask.

This is less geometrically correct than calibrated rectified stereo, but it is much easier to tune for a quick face/hand depth mask.

```bash
python live_raw_depth_mask.py \
  --left-camera 0 \
  --right-camera 2 \
  --width 640 \
  --height 480 \
  --baseline-mm 300 \
  --target-distance-mm 525 \
  --distance-window-mm 350 \
  --hfov-deg 60
```

Controls:

- `s` saves `mask.png`, `masked_left.png`, and transparent `cutout_alpha.png` into `raw_depth_mask_output/`.
- `q` quits.
- `j` / `l` shift the expected match left/right.
- `i` / `k` increase/decrease vertical search tolerance.
- `u` / `o` move the target distance nearer/farther.

If the mask is empty, try:

```bash
python live_raw_depth_mask.py --left-camera 0 --right-camera 2 --shift-sign 1
```

If it is too slow:

```bash
python live_raw_depth_mask.py --left-camera 0 --right-camera 2 --match-width 120 --shift-step-px 24
```

If it is too noisy:

```bash
python live_raw_depth_mask.py --left-camera 0 --right-camera 2 --cost-threshold 25 --min-area 1500
```

## Point Cloud Export

The older point-cloud live view is still available:

```bash
python live_reconstruct.py --calibration stereo_calibration.npz --left-camera 0 --right-camera 2 --width 640 --height 480
```

Controls:

- `s` exports a colored PLY point cloud into `recon_output/`.
- `q` quits.

## Optional Single-Camera Landmark Mesh

These scripts are optional and are not true stereo reconstruction. They use MediaPipe to create a simple landmark mesh from one image or one camera.

Install the optional dependency only if you want them:

```bash
python -m pip install -r requirements_face_hand.txt
```

Run:

```bash
python reconstruct_face_hand.py --image face_or_hand.jpg --part auto --output body_mesh.obj
python live_face_hand_mesh.py --camera 0 --width 640 --height 480 --part auto
```

## Camera Discovery

Find camera indices:

```bash
python list_cameras.py --max-index 8 --api avfoundation
```

## Notes

- The cameras must stay rigidly mounted after calibration. Any movement invalidates the calibration.
- The physically left camera should be `--left-camera`; the physically right camera should be `--right-camera`.
- Stereo needs texture. Blank skin, reflective objects, flat walls, and motion blur create holes or noisy triangles.
- Webcams are not hardware-synchronized, so static or slow face/hand poses work best.
- Mesh units are millimeters only if the chessboard square size was measured and entered correctly.

## Files

- `live_stereo_mesh.py` creates a real-time mesh from the two calibrated camera feeds.
- `stereo_mesh_common.py` converts stereo depth maps into triangle meshes and writes OBJ/PLY.
- `live_reconstruct.py` runs live stereo disparity and exports PLY point-cloud snapshots.
- `reconstruct_pair.py` builds a PLY point cloud from one stereo image pair.
- `list_cameras.py` probes OpenCV camera indices.
- `make_chessboard_svg.py` creates a printable calibration board.
- `capture_calibration_pairs.py` captures stereo chessboard image pairs.
- `calibrate_stereo.py` estimates stereo calibration and rectification.
- `stereo_common.py` contains shared capture, calibration, disparity, and PLY helpers.
- `reconstruct_face_hand.py`, `live_face_hand_mesh.py`, and `body_mesh_common.py` are optional single-camera MediaPipe landmark utilities.
