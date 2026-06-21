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
import tempfile

from FastCGHNet import FastCGHNet
from PLM import CGHGenerator, DeviceLibrary
from plmcontroller import (
    CONNECTION_LABELS,
    CONNECTION_TYPES,
    DISPLAY_FRAME_RATE,
    DLL_PATH,
    HOLOGRAMS_PER_FRAME,
    PLAY_MODES,
    PORT_SWAPS,
    configure_plm,
    load_plm_controller_class,
    plmctrl_runtime,
)

PROJECT_DIR = Path(__file__).resolve().parent


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
        "--max-frames",
        type=int,
        default=64,
        help="Maximum PLMCtrl frame slots to allocate",
    )
    parser.add_argument(
        "--x0",
        type=int,
        default=1920,
        help="X coordinate of the PLM display window (default: 1920)",
    )
    parser.add_argument(
        "--y0",
        type=int,
        default=0,
        help="Y coordinate of the PLM display window (default: 0)",
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
        "--predictor",
        choices=("cnn", "adam"),
        default="cnn",
        help="Hologram predictor: 'cnn' uses FastCGHNet MLP (fast, real-time); 'adam' uses iterative ADAM optimization (slower, higher quality). Default: cnn",
    )
    parser.add_argument(
        "--model-path",
        default=PROJECT_DIR / "models" / "best_model.pt",
        help="Path to the FastCGHNet model checkpoint (used with --predictor cnn)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("adam", "adamwgs"),
        default="adamwgs",
        help="Iterative algorithm to use for hologram generation (used with --predictor adam; default: adamwgs)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Number of iterations for hologram optimization (used with --predictor adam; default: 50)",
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
        choices=tuple(PORT_SWAPS.keys()),
        default="bac",
        help="PLM input port swap",
    )
    parser.add_argument(
        "--upload-mode",
        choices=("packed-bitmap", "gpu-phase"),
        default="packed-bitmap",
        help=(
            "PLM upload mode. packed-bitmap packs the PLM-ready binary CGH into "
            "all 24 RGB bitplanes; gpu-phase lets plmctrl bitpack normalized phase."
        ),
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
        "--ui-warmup-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait after StartUI before uploading packed frames",
    )
    parser.add_argument(
        "--pre-play-delay-seconds",
        type=float,
        default=0.5,
        help="Seconds to wait after the first upload before PLM play()",
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
        help="Do not call PLM play() after uploading the first frame",
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


def load_cnn_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load FastCGHNet CNN model from checkpoint."""
    if not model_path.is_absolute() and not model_path.exists():
        model_path = PROJECT_DIR / model_path
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


def initialize_generator() -> tuple[CGHGenerator, DeviceLibrary, dict]:
    """Initialize the iterative hologram generator and device configuration."""
    generator = CGHGenerator()
    device_lib = DeviceLibrary()
    device_dict = device_lib.defineDevice("0.67")
    return generator, device_lib, device_dict


def predict_hologram_cnn(
    frame_image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    generator: CGHGenerator,
    device_lib: DeviceLibrary,
    device_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate hologram using FastCGHNet CNN (fast, real-time)."""
    tensor = torch.from_numpy(frame_image[np.newaxis, np.newaxis, :, :]).float().to(device)
    with torch.no_grad():
        phase_pred = model(tensor)

    phase_np = phase_pred.squeeze().cpu().numpy()
    _, state_disc = generator.discretePhase(
        phase_np,
        device_dict["nLevel"],
        device_dict["pLevel"],
    )
    cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
    return phase_np.astype(np.float32, copy=False), cgh_mapped.astype(np.uint8)


def predict_hologram_adam(
    frame_image: np.ndarray,
    generator: CGHGenerator,
    device_dict: dict,
    algorithm: str,
    num_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate hologram using iterative optimization (ADAM or ADAMWGS)."""
    # Save frame to temporary file for loading into generator
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        tmp_path = tmp.name
        Image.fromarray((frame_image * 255).astype(np.uint8), mode="L").save(tmp_path)
    
    try:
        # Run iterative hologram generation
        generator.createCGH(
            DeviceDictionary=device_dict,
            filename=tmp_path,
            alg=algorithm.upper(),
            numIter=num_iter,
            propMethod="Fourier",
            binarizeTarget=False,
            lossMode="auto",
        )
        phase = generator.CGH_output_phase_disc
        cgh_mapped = generator.CGH_mapped
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return phase.astype(np.float32, copy=False), cgh_mapped.astype(np.uint8)


def pack_repeated_bitplane_frame(cgh_mapped: np.ndarray) -> np.ndarray:
    """Pack one binary PLM-ready CGH into all 24 RGB temporal bitplanes."""
    h, w = cgh_mapped.shape
    bitplane = (cgh_mapped > 127).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255

    for bit_index in range(HOLOGRAMS_PER_FRAME):
        channel = bit_index // 8
        bit_offset = bit_index % 8
        rgba[:, :, channel] |= bitplane << bit_offset

    return np.ascontiguousarray(rgba.reshape(h, w * 4))


def phase_to_plm_units(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim != 2:
        raise ValueError("phase must be a 2D array")

    phase = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
    return np.ascontiguousarray(phase, dtype=np.float32)


def repeated_phase_stack(phase: np.ndarray) -> np.ndarray:
    phase_plm = phase_to_plm_units(phase)
    return np.ascontiguousarray(
        np.repeat(phase_plm[np.newaxis, :, :], HOLOGRAMS_PER_FRAME, axis=0),
        dtype=np.float32,
    )


def upload_hologram_frame(
    plm,
    phase: np.ndarray,
    cgh_mapped: np.ndarray,
    upload_mode: str,
) -> None:
    if upload_mode == "gpu-phase":
        result = plm.bitpack_and_insert_gpu(repeated_phase_stack(phase), 0)
    else:
        frame = pack_repeated_bitplane_frame(cgh_mapped)
        result = plm.insert_frames(frame, 0, format=1)
        plm.set_frame(0)

    if result == 0:
        raise RuntimeError("Failed to upload PLM hologram frame")


def save_training_frame(output_dir: Path, frame_index: int, binary_image: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"training_frame_{frame_index:04d}.bmp"
    saved = Image.fromarray((binary_image * 255).astype(np.uint8), mode="L")
    saved.save(filename, format="BMP")


def configure_plm_controller(plm, args: argparse.Namespace) -> int:
    connection = args.connection
    if connection == "auto":
        connection = "hdmi"

    connection_type = CONNECTION_TYPES[connection]
    play_mode = PLAY_MODES[args.play_mode]
    port_swap = PORT_SWAPS[args.port_swap]

    if args.windowed or (not args.exclusive_fullscreen):
        print("Using windowed DirectX swapchain for the PLM UI.")
        plm.set_windowed(True)
    else:
        print("Using exclusive/fullscreen DirectX swapchain for the PLM UI.")
        plm.set_windowed(False)

    configure_plm(plm, play_mode, connection_type, port_swap, args.settle_seconds)
    print(f"Starting PLM UI at ({args.x0}, {args.y0})...")
    plm.start_ui()
    time.sleep(max(0.0, args.ui_warmup_seconds))
    return connection_type


def run_pipeline(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize generator and device for both predictor modes
    generator, device_lib, device_dict = initialize_generator()
    
    # Initialize predictor based on choice
    if args.predictor == "cnn":
        model = load_cnn_model(Path(args.model_path), device)
    else:  # adam
        model = None

    if args.mode == "training" and args.save_training_dir:
        args.save_training_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.camera_index, args.camera_width, args.camera_height)
    print(f"Opened camera {args.camera_index} ({args.camera_width}x{args.camera_height})")
    print(f"Pipeline mode: {args.mode}")
    print(f"PLM upload mode: {args.upload_mode}")
    if args.predictor == "cnn":
        print(f"Using predictor: FastCGHNet CNN (MLP)")
        print(f"Model path: {Path(args.model_path).resolve()}")
    else:
        print(f"Using predictor: Iterative {args.algorithm.upper()}")
        print(f"Iterations: {args.num_iterations}")

    with plmctrl_runtime():
        PLMController = load_plm_controller_class()
        plm = PLMController(
            args.max_frames,
            args.plm_width,
            args.plm_height,
            dll_path=str(DLL_PATH),
            x0=args.x0,
            y0=args.y0,
        )

        if args.connection == "auto":
            args.connection = "hdmi"

        connection_type = configure_plm_controller(plm, args)
        rgb_frame_rate = DISPLAY_FRAME_RATE[connection_type]
        bitplane_rate = rgb_frame_rate * HOLOGRAMS_PER_FRAME
        print(
            f"Prepared {CONNECTION_LABELS[connection_type]} packed playback: "
            f"{rgb_frame_rate:g} Hz x {HOLOGRAMS_PER_FRAME} bitplanes = "
            f"{bitplane_rate:g} Hz."
        )

        frame_index = 0
        playback_started = False
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

                if args.predictor == "cnn":
                    phase, cgh_mapped = predict_hologram_cnn(
                        processed,
                        model,
                        device,
                        generator,
                        device_lib,
                        device_dict,
                    )
                else:
                    phase, cgh_mapped = predict_hologram_adam(
                        processed,
                        generator,
                        device_dict,
                        args.algorithm,
                        args.num_iterations,
                    )

                upload_hologram_frame(plm, phase, cgh_mapped, args.upload_mode)

                if not args.no_play and not playback_started:
                    if args.pre_play_delay_seconds > 0:
                        time.sleep(args.pre_play_delay_seconds)
                    print("Starting PLM playback...")
                    plm.play()
                    playback_started = True

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
            if playback_started:
                plm.stop()
            plm.stop_ui()

    return 0


if __name__ == "__main__":
    parsed_args = parse_args()
    raise SystemExit(run_pipeline(parsed_args))