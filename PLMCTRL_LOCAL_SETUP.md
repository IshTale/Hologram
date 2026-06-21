# Local PLMCtrl Setup

This folder is configured to use the Python wrapper and DLL from:

```text
plmctrl-main/wrappers/PLMController.py
plmctrl-main/bin/plmctrl.dll
plmctrl-main/bin/BitpackHologramsCS.hlsl
```

The PLM video input configuration sent by `plmcontroller.py` uses `Parallel RGB`, `24-bit`, input port data swap `ABC -> ABC`, and HDMI by default. Use `--port-swap bac` only if your hardware setup was specifically verified with `ABC -> BAC`.

Normal frame-sequence mode uses a windowed DirectX swapchain by default because that path initializes reliably during testing. Packed-bitplane mode uses the fullscreen/exclusive swapchain path by default for real PLM timing; pass `--windowed` only for debugging.

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify that Python can load the wrapper and DLL without touching the PLM:

```powershell
python .\plmcontroller.py --check-only
```

## Normal frame sequence

Display the first four built-in Bear TMUX frames as normal full frames, with no bitplane packing:

```powershell
python .\plmcontroller.py
```

By default, normal mode uses frames `000` through `003`, uploads each image into its own PLMCtrl frame slot, and loops them through the PLMCtrl VSync-paced sequence player. This is video-frame playback, not PLM bitplane playback: expect about `30 fps` over HDMI or `60 fps` over DisplayPort.

For normal `60 fps` playback, use DisplayPort:

```powershell
python .\plmcontroller.py --connection displayport
```

Include all five built-in frames:

```powershell
python .\plmcontroller.py --sequence-image-count 0
```

For a slow visible check, use manual advance mode:

```powershell
python .\plmcontroller.py --advance-mode manual --image-duration 1
```

## Packed bitplane playback

Display the first four built-in Bear TMUX frames packed into one 24-bitplane RGB frame over HDMI:

```powershell
python .\plmcontroller.py --playback-mode bitplane-package
```

Packed mode uses frames `000` through `003`, repeats them evenly across the 24 RGB bitplanes as `0, 1, 2, 3` six times, uploads one packed frame into PLMCtrl slot `0`, and uses HDMI timing: `30 Hz x 24 bitplanes = 720 Hz`.

Packed mode expects the input BMPs to already be binary `0/255`, `2716 x 1600`, PLM-ready bitplane images. It does not generate valid phase holograms from grayscale photos or phase maps. If you deliberately want the old thresholding behavior, add `--allow-threshold-packed-images`.

If your hardware actually needs `ABC -> BAC`, add the port swap option:

```powershell
python .\plmcontroller.py --playback-mode bitplane-package --port-swap bac
```

Packed mode will not print `Displaying image 1/5` style messages because Python is not stepping through separate frames. The PLM reads the packed bitplanes inside one RGB frame.

Use grouped bitplanes instead, six copies of each image:

```powershell
python .\plmcontroller.py --playback-mode bitplane-package --bitplane-layout grouped
```

If you switch to DisplayPort for 1440 Hz, run:

```powershell
python .\plmcontroller.py --playback-mode bitplane-package --connection displayport
```

DisplayPort timing is `60 Hz x 24 bitplanes = 1440 Hz`. If `SetVideoPatternMode` fails in DisplayPort mode, the PLM is usually not ready on DisplayPort. Close DLP LightCrafter, confirm the DisplayPort cable is connected and the PLM appears as an active Windows display, then retry.

Start the PLMCtrl UI after configuration:

```powershell
python .\plmcontroller.py --start-ui
```

For a test window instead of full-screen output:

```powershell
python .\plmcontroller.py --start-ui --windowed
```

Display custom image files sequentially and repeat forever:

```powershell
python .\plmcontroller.py --images .\image1.png .\image2.png .\image3.png .\image4.png --sequence-image-count 0
```

For a debug window while testing the image sequence:

```powershell
python .\plmcontroller.py --windowed
```

For a slow visual check in the debug window, use manual advance mode:

```powershell
python .\plmcontroller.py --windowed --advance-mode manual --image-duration 1
```

The built-in sequence uses these five files:

```text
C:\Users\areeb\OneDrive\Desktop\Unchanged_tiplmsuite-release-1.0.0.4\temporal_mux_output\cgh_frames\Bear1_TO_Bear2_TMUX_frame_000.bmp
C:\Users\areeb\OneDrive\Desktop\Unchanged_tiplmsuite-release-1.0.0.4\temporal_mux_output\cgh_frames\Bear1_TO_Bear2_TMUX_frame_001.bmp
C:\Users\areeb\OneDrive\Desktop\Unchanged_tiplmsuite-release-1.0.0.4\temporal_mux_output\cgh_frames\Bear1_TO_Bear2_TMUX_frame_002.bmp
C:\Users\areeb\OneDrive\Desktop\Unchanged_tiplmsuite-release-1.0.0.4\temporal_mux_output\cgh_frames\Bear1_TO_Bear2_TMUX_frame_003.bmp
C:\Users\areeb\OneDrive\Desktop\Unchanged_tiplmsuite-release-1.0.0.4\temporal_mux_output\cgh_frames\Bear1_TO_Bear2_TMUX_frame_004.bmp
```

Full-size PLM frames, such as these `2716 x 1600` BMPs, are uploaded directly into PLMCtrl frame slots. Other image sizes are converted to grayscale phase maps, resized to the PLM resolution, bitpacked, then uploaded.
