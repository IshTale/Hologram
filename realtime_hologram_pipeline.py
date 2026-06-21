#!/usr/bin/env python3
"""
Real-time camera-to-PLM pipeline.

This script opens a camera feed, processes each frame, and sends a predicted
hologram to the TI PLM via plmcontroller. The user can choose between two
processing modes:

- training: convert the camera frame into a binary training-style image,
  then predict a hologram from that binary image.
- direct: use the camera frame directly as input to the hologram predictor.

The resulting PLM frame is uploaded continuously for real-time display.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from FastCGHNet import FastCGHNet
from PLM import CGHGenerator, DeviceLibrary
from plmcontroller import (
    CONNECTION_TYPES,
    DLL_PATH,
    PLAY_MODES,
    configure_plm,
    load_plm_controller_class,
    plmctrl_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time camera feed -> hologram -> PLM display pipeline"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use (default: 0)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested camera capture width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested camera capture height",
    )
    parser.add_argument(
        "--plm-width",
        type=int,
        default=1358,
        help="PLM frame width (default: 1358)",
    )
    parser.add_argument(
        "--plm-height",
        type=int,
        default=800,
        help="PLM frame height (default: 800)",
    )
    parser.add_argument(
        "--mode",
        choices=("training", "direct"),
        default="direct",
        help=(
            "Processing mode: both modes quantize the frame before prediction. "
            "Use 'training' to also save the binary training-style frames; 'direct' skips saving."
        ),
    )
    parser.add_argument(
        "--model-path",
        default="./models/best_model.pt",
        help="Path to the FastCGHNet model checkpoint",
    )
    parser.add_argument(
        "--save-training-dir",
        type=Path,
        default=None,
        help="Optional directory to save binary training-style frames when using training mode",
    )
    parser.add_argument(
        "--connection",
        choices=("auto", "hdmi", "displayport", "dp"),
        default="hdmi",
        help="Video connection type for PLM configuration",
    )
    parser.add_argument(
        "--play-mode",
        choices=tuple(PLAY_MODES.keys()),
        default="continuous",
        help="PLM play mode",
    )
    parser.add_argument(
        "--port-swap",
        choices=("abc", "bac"),
        default="abc",
        help="PLM input port swap",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Use windowed swapchain for PLM UI",
    )
    parser.add_argument(
        "--exclusive-fullscreen",
        action="store_true",
        help="Use exclusive/fullscreen swapchain for PLM UI",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait after configuring PLM before starting UI",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.2,
        help="Seconds between processing camera frames",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview windows for camera, training, and hologram data",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Process a single camera frame and exit",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not call PLM play() after uploading frames",
    )
    return parser.parse_args()


def quantize_to_1bit(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    image = image.copy()
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            old_val = image[y, x]
            new_val = 1.0 if old_val >= 0.5 else 0.0
            error = old_val - new_val
            image[y, x] = new_val

            if x + 1 < width:
                image[y, x + 1] += error * 7.0 / 16.0
            if y + 1 < height:
                if x - 1 >= 0:
                    image[y + 1, x - 1] += error * 3.0 / 16.0
                image[y + 1, x] += error * 5.0 / 16.0
                if x + 1 < width:
                    image[y + 1, x + 1] += error * 1.0 / 16.0

    return (image > 0.5).astype(np.uint8)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def preprocess_frame(frame: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    quantized = quantize_to_1bit(resized.astype(np.float32) / 255.0).astype(np.float32)
    return quantized


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = FastCGHNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_hologram(frame_image: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(frame_image[np.newaxis, np.newaxis, :, :]).float().to(device)
    with torch.no_grad():
        phase_pred = model(tensor)

    phase_np = phase_pred.squeeze().cpu().numpy()
    device_lib = DeviceLibrary()
    device_dict = device_lib.defineDevice("0.67")
    _, state_disc = CGHGenerator.discretePhase(
        phase_np,
        device_dict["nLevel"],
        device_dict["pLevel"],
    )
    cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
    return cgh_mapped.astype(np.uint8)


def cgh_to_rgba_frame(cgh_mapped: np.ndarray) -> np.ndarray:
    h, w = cgh_mapped.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = np.expand_dims(cgh_mapped, axis=2)
    rgba[:, :, 3] = 255
    return np.ascontiguousarray(rgba.reshape(h, w * 4))


def upload_hologram_frame(plm, cgh_mapped: np.ndarray) -> None:
    frame = cgh_to_rgba_frame(cgh_mapped)
    result = plm.insert_frames(frame, 0, format=1)
    if result == 0:
        raise RuntimeError("Failed to upload PLM hologram frame")
    plm.set_frame(0)


def save_training_frame(output_dir: Path, frame_index: int, binary_image: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"training_frame_{frame_index:04d}.bmp"
    saved = Image.fromarray((binary_image * 255).astype(np.uint8), mode="L")
    saved.save(filename, format="BMP")


def configure_plm_controller(plm, args: argparse.Namespace) -> None:
    connection = args.connection
    if connection == "auto":
        connection = "hdmi"

    connection_type = CONNECTION_TYPES[connection]
    play_mode = PLAY_MODES[args.play_mode]
    port_swap = 0 if args.port_swap == "abc" else 4

    if args.windowed or (not args.exclusive_fullscreen):
        plm.set_windowed(True)
    else:
        plm.set_windowed(False)

    configure_plm(plm, play_mode, connection_type, port_swap, args.settle_seconds)
    plm.start_ui()
    time.sleep(max(0.5, args.settle_seconds))


def run_pipeline(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path), device)

    if args.mode == "training" and args.save_training_dir:
        args.save_training_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.camera_index, args.camera_width, args.camera_height)
    print(f"Opened camera {args.camera_index} ({args.camera_width}x{args.camera_height})")
    print(f"Pipeline mode: {args.mode}")
    print(f"Using model: {Path(args.model_path).resolve()}")

    with plmctrl_runtime():
        PLMController = load_plm_controller_class()
        plm = PLMController(
            64,
            args.plm_width,
            args.plm_height,
            dll_path=str(DLL_PATH),
            x0=0,
            y0=0,
        )

        if args.connection == "auto":
            args.connection = "hdmi"

        configure_plm_controller(plm, args)

        if not args.no_play:
            print("Starting PLM playback...")
            plm.play()

        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: camera frame read failed")
                    break

                processed = preprocess_frame(
                    frame,
                    args.plm_width,
                    args.plm_height,
                    args.mode,
                )

                if args.mode == "training" and args.save_training_dir:
                    save_training_frame(args.save_training_dir, frame_index, processed)

                cgh_mapped = predict_hologram(processed, model, device)
                upload_hologram_frame(plm, cgh_mapped)

                if args.preview:
                    cv2.imshow("Camera input", frame)
                    cv2.imshow(
                        "Processed input",
                        (processed * 255).astype(np.uint8) if processed.dtype != np.uint8 else processed,
                    )
                    cv2.imshow("PLM hologram preview", cgh_mapped)
                    if cv2.waitKey(max(1, int(args.frame_interval * 1000))) & 0xFF == ord("q"):
                        break

                frame_index += 1
                if args.run_once:
                    break

                time.sleep(args.frame_interval)

        except KeyboardInterrupt:
            print("Stopping real-time pipeline...")

        finally:
            cap.release()
            if args.preview:
                cv2.destroyAllWindows()
            if not args.no_play:
                plm.stop()
            plm.stop_ui()

    return 0


if __name__ == "__main__":
    parsed_args = parse_args()
    raise SystemExit(run_pipeline(parsed_args))
