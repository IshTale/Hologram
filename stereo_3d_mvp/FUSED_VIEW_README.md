# Stereo → one RGB-D "3D reconstruction mask" → hologram

`fuse_stereo_view.py` turns the two camera feeds into a **single depth-aware view** the
hologram stage can consume: reference RGB + dense metric depth + subject alpha, plus a
ready-to-use `hologram_input` (1358×800).

## Run

```bash
cd stereo_3d_mvp && source .venv/bin/activate

# live (your rig)
python fuse_stereo_view.py --left-camera 0 --right-camera 2 \
  --baseline-mm 300 --target-distance-mm 525 --distance-window-mm 350 --hfov-deg 60

# offline (two saved frames — testable without the rig)
python fuse_stereo_view.py --left-image L.png --right-image R.png --output-dir out
```

Live keys: `s` save bundle · `q` quit · `j/l` x-shift · `u/o` distance · `[ ]` matte threshold.

### Output bundle (`*_rgbd.npz` + PNGs)
`rgb`, `depth_mm` (metric), `depth_norm` (**[0,1] z-squeezed over [near,far] — the CGH input format**),
`alpha` (subject matte), `hologram_input` (grayscale subject-on-black, 1358×800), `cutout` (RGBA).

## How it works (and why)

The rig is **wide-baseline / short-range**: 300 mm baseline at ~525 mm ⇒ `b/Z ≈ 0.57`,
max disparity ≈ **317 px @525 mm / 475 px @350 mm** (half-to-¾ of a 640 frame), severe
occlusion. So stereo can only give a coarse near/far *band*, never clean dense face depth.
The design therefore is **segmentation-led, stereo-anchored**:

1. **Matte** — MediaPipe `ImageSegmenter` (Tasks API, model auto-downloaded) on the left feed.
2. **Depth** — dense raw-shift disparity → metric depth (both cameras).
3. **Fuse** — depth is a *component-selection gate* (pick the depth-supported subject blob,
   don't multiply the matte by the noisy band), then `ximgproc.guidedFilter` snaps edges.
4. **Emit** — left RGB + matte + z-squeezed `depth_norm`.

## Rig verdict (from the literature review)

**300 mm at 525 mm is too wide.** Recommended baseline ≈ **75–80 mm** (`b/Z ≈ 0.15`,
max disparity ~83 px @525 mm), which keeps cameras near fronto-parallel so rectification
behaves and SGBM works. Narrowing the baseline is the single highest-leverage hardware fix.
The pipeline above runs on the current 300 mm rig today and gets strictly better at 75 mm.

## Roadmap (research-backed)

**Now (pure OpenCV, shipped):** segmentation matte + depth selection-gate + guided-filter edges
+ z-squeezed `depth_norm`. ← `fuse_stereo_view.py`

**Next (needs `pip install onnxruntime`; torch is absent):**
1. **Depth Anything V2 (ViT-S, ONNX)** dense mono depth on the left frame; align to metric by a
   **RANSAC least-squares scale+shift in disparity** over confident SGBM points in the face ROI
   (`D_fused = c·d_stereo + (1−c)·d_mono`, mono fills occlusions). EMA-smooth `s,t`.
2. **Robust Video Matting (RVM)** ONNX (recurrent, better hair/temporal stability) to replace MediaPipe.
3. **Two-view DIBR center view**: forward-warp both cameras to a virtual center pose (z-buffer /
   softmax splat color+depth), blend (each camera fills the other's disocclusions), then
   background-only `INPAINT_NS` for residual holes.

**Hologram input:** layer-based / Tensor-Holography CGH wants **RGB + linear depth in [0,1] over a
THIN volume** (already emitted as `depth_norm`). Do **not** drop MIT pretrained weights on the TI
0.67 PLM (different pitch/λ/phase-levels) — validate with classical layer-based ASM CGH, then
retrain the image→phase U-Net on data from your own simulator (2–8 depth planes).

## Key references
- Depth Anything V2 / MiDaS scale-shift alignment — https://ar5iv.labs.arxiv.org/html/2307.14460
- Robust Video Matting (RVM) — https://github.com/PeterL1n/RobustVideoMatting
- Softmax Splatting (Niklaus & Liu, CVPR 2020), forward warp — https://sniklaus.com/softsplat
- 3D-Photo Layered Depth Inpainting (Shih et al., CVPR 2020) — https://shihmengli.github.io/3D-Photo-Inpainting/
- Wide-Baseline Novel View Synthesis (Du et al., CVPR 2023) — https://yilundu.github.io/wide_baseline/
- Tensor Holography (Shi et al., Nature 2021) — https://www.nature.com/articles/s41586-020-03152-0
- Tensor Holography V2 (Shi et al., 2022) — https://www.nature.com/articles/s41377-022-00894-6
