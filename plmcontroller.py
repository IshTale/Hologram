"""Configure and start a TI PLM through structuredlightlab/plmctrl.

The upstream Python wrapper and DLL live in ./plmctrl-main. This launcher
keeps those paths explicit so the script can be run from this project folder
without copying DLLs into the root.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
import time
from types import ModuleType
from typing import Iterator, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
PLMCTRL_DIR = PROJECT_DIR / "plmctrl-main"
WRAPPER_PATH = PLMCTRL_DIR / "wrappers" / "PLMController.py"
BIN_DIR = PLMCTRL_DIR / "bin"
DLL_PATH = BIN_DIR / "plmctrl.dll"
SHADER_PATH = BIN_DIR / "BitpackHologramsCS.hlsl"

CONNECTION_TYPES = {
    "hdmi": 1,
    "displayport": 2,
    "dp": 2,
}
CONNECTION_LABELS = {
    1: "HDMI",
    2: "DisplayPort",
}

PLAY_MODES = {
    "once": 0,
    "continuous": 1,
}

HOLOGRAMS_PER_FRAME = 24
PORT_SWAPS = {
    "abc": 0,
    "bac": 4,
}
PORT_SWAP_LABELS = {
    0: "ABC -> ABC",
    4: "ABC -> BAC",
}
PACKED_BITPLANE_COUNT = 24
DEFAULT_SEQUENCE_IMAGE_COUNT = 4
DISPLAY_FRAME_RATE = {
    1: 30.0,  # HDMI: 30 RGB frames/s * 24 bitplanes = 720 Hz
    2: 60.0,  # DisplayPort: 60 RGB frames/s * 24 bitplanes = 1440 Hz
}
DEFAULT_IMAGE_SEQUENCE = (
    Path(
        r"C:\Users\areeb\OneDrive\Desktop"
        r"\Unchanged_tiplmsuite-release-1.0.0.4"
        r"\temporal_mux_output\cgh_frames"
        r"\Bear1_TO_Bear2_TMUX_frame_000.bmp"
    ),
    Path(
        r"C:\Users\areeb\OneDrive\Desktop"
        r"\Unchanged_tiplmsuite-release-1.0.0.4"
        r"\temporal_mux_output\cgh_frames"
        r"\Bear1_TO_Bear2_TMUX_frame_001.bmp"
    ),
    Path(
        r"C:\Users\areeb\OneDrive\Desktop"
        r"\Unchanged_tiplmsuite-release-1.0.0.4"
        r"\temporal_mux_output\cgh_frames"
        r"\Bear1_TO_Bear2_TMUX_frame_002.bmp"
    ),
    Path(
        r"C:\Users\areeb\OneDrive\Desktop"
        r"\Unchanged_tiplmsuite-release-1.0.0.4"
        r"\temporal_mux_output\cgh_frames"
        r"\Bear1_TO_Bear2_TMUX_frame_003.bmp"
    ),
    Path(
        r"C:\Users\areeb\OneDrive\Desktop"
        r"\Unchanged_tiplmsuite-release-1.0.0.4"
        r"\temporal_mux_output\cgh_frames"
        r"\Bear1_TO_Bear2_TMUX_frame_004.bmp"
    ),
)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def resolve_input_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def load_plm_controller_class():
    require_file(WRAPPER_PATH, "PLM Python wrapper")

    spec = importlib.util.spec_from_file_location("plmctrl_python_wrapper", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load wrapper module from {WRAPPER_PATH}")

    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)
    return module.PLMController


@contextmanager
def plmctrl_runtime() -> Iterator[None]:
    """Prepare DLL search paths and cwd for plmctrl.

    The C++ library compiles BitpackHologramsCS.hlsl by filename, so GPU/UI
    functions need the current working directory to be the bin directory.
    """

    require_file(DLL_PATH, "plmctrl DLL")
    require_file(SHADER_PATH, "plmctrl compute shader")

    previous_cwd = Path.cwd()
    dll_directory_handle = None

    if hasattr(os, "add_dll_directory"):
        dll_directory_handle = os.add_dll_directory(str(BIN_DIR))
    else:
        os.environ["PATH"] = f"{BIN_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

    os.chdir(BIN_DIR)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if dll_directory_handle is not None:
            dll_directory_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure the PLM using the local plmctrl wrapper and DLL."
    )
    parser.add_argument(
        "--playback-mode",
        choices=("bitplane-package", "frame-sequence"),
        default="frame-sequence",
        help=(
            "Use normal full-frame sequence playback or packed bitplane playback. "
            "The default is frame-sequence."
        ),
    )
    parser.add_argument(
        "--connection",
        choices=("auto", *CONNECTION_TYPES.keys()),
        default="hdmi",
        help="Video connection to configure. HDMI is the default for this setup.",
    )
    parser.add_argument("--play-mode", choices=PLAY_MODES.keys(), default="continuous")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--width", type=int, default=1358)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--x0", type=int, default=1920)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="Delay after SetConnectionType and SetVideoPatternMode.",
    )
    parser.add_argument(
        "--start-ui",
        action="store_true",
        help="Start the plmctrl display UI after configuring the PLM.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help=(
            "Use a windowed DirectX swapchain. Useful for testing; frame-sequence "
            "mode uses this by default."
        ),
    )
    parser.add_argument(
        "--exclusive-fullscreen",
        action="store_true",
        help=(
            "Use the fullscreen/exclusive swapchain path. Packed-bitplane mode "
            "uses this by default."
        ),
    )
    parser.add_argument(
        "--port-swap",
        choices=PORT_SWAPS.keys(),
        default="abc",
        help=(
            "Input port data swap. Default abc matches the plmctrl README "
            "configuration; use bac only if your hardware was verified that way."
        ),
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Configure the PLM but do not call Play().",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify that the wrapper and DLL can be loaded.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="IMAGE",
        help="Image files to display. Overrides the built-in Bear sequence.",
    )
    parser.add_argument(
        "--no-default-images",
        action="store_true",
        help="Do not display the built-in Bear image sequence.",
    )
    parser.add_argument(
        "--image-duration",
        type=float,
        default=1.0,
        help="Seconds to show each image in manual advance mode.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of full sequence cycles to run. Use 0 to repeat forever.",
    )
    parser.add_argument(
        "--advance-mode",
        choices=("plmctrl", "manual"),
        default="plmctrl",
        help="For frame-sequence mode: use PLMCtrl's VSync-paced player or a slow Python frame loop.",
    )
    parser.add_argument(
        "--sequence-image-count",
        type=int,
        default=DEFAULT_SEQUENCE_IMAGE_COUNT,
        help=(
            "For frame-sequence mode: number of selected images to use. "
            "Use 0 to display all selected images."
        ),
    )
    parser.add_argument(
        "--packed-image-count",
        type=int,
        default=4,
        help="Number of images to pack into the 24 RGB bitplanes.",
    )
    parser.add_argument(
        "--bitplane-layout",
        choices=("interleaved", "grouped"),
        default="interleaved",
        help="Interleaved repeats 0,1,2,3 across bitplanes; grouped uses six copies of each image.",
    )
    parser.add_argument(
        "--allow-threshold-packed-images",
        action="store_true",
        help=(
            "Allow packed-bitplane mode to threshold non-binary images. By default "
            "packed mode requires 0/255 PLM-ready bitplane BMPs."
        ),
    )
    parser.add_argument(
        "--sequence-fps",
        type=float,
        default=None,
        help=(
            "Display refresh rate used to time frame-sequence repeats. Default is "
            "based on the configured connection: HDMI 30 fps, DisplayPort 60 fps."
        ),
    )
    parser.add_argument(
        "--ui-warmup-seconds",
        type=float,
        default=1.0,
        help="Delay after StartUI before GPU bitpacking/upload.",
    )
    return parser.parse_args()


def configure_plm(
    plm,
    play_mode: int,
    connection_type: int,
    port_swap: int,
    settle_seconds: float,
) -> None:
    connection_label = CONNECTION_LABELS[connection_type]
    port_swap_label = PORT_SWAP_LABELS.get(port_swap, str(port_swap))

    print("Opening PLM USB connection...")
    plm.open()

    print("Setting source: Parallel RGB, 24-bit")
    plm.set_source(0, 1)

    print(f"Setting port swap: {port_swap_label}")
    plm.set_port_swap(0, port_swap)
    plm.set_port_swap(1, port_swap)

    print(f"Setting pixel mode: {connection_label}")
    plm.set_pixel_mode(connection_type)

    print(f"Setting connection type: {connection_label}")
    plm.set_connection_type(connection_type)
    time.sleep(settle_seconds)

    try:
        plm.set_video_pattern_mode()
    except RuntimeError as exc:
        hint = (
            "DisplayPort is required for 1440 Hz packed bitplane playback. "
            "Make sure the PLM is connected over DisplayPort, Windows sees it as an active "
            "display, and DLP LightCrafter is closed. If you are currently using HDMI, run "
            "`python .\\plmcontroller.py --connection hdmi`; that keeps packed bitplanes "
            "working but uses HDMI timing, 30 Hz x 24 = 720 Hz."
        )
        if connection_type == CONNECTION_TYPES["hdmi"]:
            hint = (
                "Make sure DLP LightCrafter is closed and the PLM is connected over HDMI "
                "as an active Windows display before running this script."
            )
        raise RuntimeError(
            f"SetVideoPatternMode failed while configuring {connection_label}. {hint}"
        ) from exc

    time.sleep(settle_seconds)

    plm.update_lut(play_mode, connection_type)


def image_to_phase(path: Path, width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for --images. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    require_file(path, "sequence image")

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((width, height), resampling)
        phase = np.asarray(image, dtype=np.float32) / 255.0

    return np.ascontiguousarray(phase, dtype=np.float32)


def image_to_direct_rgba_frame(path: Path, width: int, height: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for image loading. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    require_file(path, "sequence image")

    expected_size = (2 * width, 2 * height)
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(
                f"{path.name} is {image.size}, not the PLM frame size {expected_size}"
            )
        rgba = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)

    return np.ascontiguousarray(rgba.reshape(2 * height, 4 * 2 * width))


# Packed bitplane playback.
def image_to_binary_bitplane(
    path: Path,
    width: int,
    height: int,
    allow_threshold: bool,
) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for image loading. Install it with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    require_file(path, "packed bitplane image")

    expected_size = (2 * width, 2 * height)
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(
                f"{path.name} is {image.size}, not the PLM frame size {expected_size}"
            )
        luminance = np.array(image.convert("L"), dtype=np.uint8, copy=True)

    values = np.unique(luminance)
    if not np.all(np.isin(values, (0, 255))):
        if not allow_threshold:
            raise ValueError(
                f"{path.name} is not a binary 0/255 bitplane image. "
                "Packed-bitplane mode expects PLM-ready binary BMPs; it does not "
                "generate valid holograms from grayscale photos or phase maps. "
                "Use --allow-threshold-packed-images to threshold anyway."
            )
        print(f"Warning: thresholding non-binary packed image: {path.name}")

    return luminance > 127


def bitplane_order(image_count: int, layout: str) -> np.ndarray:
    if image_count <= 0:
        raise ValueError("At least one image is required for bitplane packaging")
    if image_count > PACKED_BITPLANE_COUNT:
        raise ValueError(f"Cannot pack more than {PACKED_BITPLANE_COUNT} images")

    if layout == "interleaved":
        return np.arange(PACKED_BITPLANE_COUNT, dtype=np.uint8) % image_count

    if layout == "grouped":
        repeats = int(np.ceil(PACKED_BITPLANE_COUNT / image_count))
        return np.repeat(np.arange(image_count, dtype=np.uint8), repeats)[
            :PACKED_BITPLANE_COUNT
        ]

    raise ValueError(f"Unknown bitplane layout: {layout}")


def pack_binary_bitplanes(bitplanes: Sequence[np.ndarray], layout: str) -> np.ndarray:
    order = bitplane_order(len(bitplanes), layout)
    height, width = bitplanes[0].shape

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255

    for bit_index, image_index in enumerate(order):
        channel = bit_index // 8
        bit_offset = bit_index % 8
        rgba[:, :, channel] |= bitplanes[int(image_index)].astype(np.uint8) << bit_offset

    return np.ascontiguousarray(rgba.reshape(height, 4 * width))


def upload_packed_bitplane_frame(
    plm,
    image_paths: Sequence[Path],
    width: int,
    height: int,
    layout: str,
    allow_threshold: bool,
) -> None:
    print(
        f"Packing {len(image_paths)} images into one "
        f"{PACKED_BITPLANE_COUNT}-bitplane RGB frame..."
    )
    print(f"Bitplane order: {bitplane_order(len(image_paths), layout).tolist()}")
    print("Packed mode expects binary 0/255 PLM-ready bitplane images.")

    bitplanes = [
        image_to_binary_bitplane(image_path, width, height, allow_threshold)
        for image_path in image_paths
    ]
    frame = pack_binary_bitplanes(bitplanes, layout)

    result = plm.insert_frames(frame, 0, format=1)
    if result == 0:
        raise RuntimeError("Failed to upload packed bitplane frame")
    plm.set_frame(0)


# Normal full-frame sequence playback.
def upload_image_sequence(
    plm,
    image_paths: Sequence[Path],
    width: int,
    height: int,
    sequence_slots: int,
) -> None:
    print(f"Uploading {len(image_paths)} image frames to PLMCtrl...")

    for index, image_path in enumerate(image_paths):
        print(f"  {index + 1}/{len(image_paths)}: {image_path.name}")

        try:
            frame = image_to_direct_rgba_frame(image_path, width, height)
            result = plm.insert_frames(frame, index, format=1)
        except ValueError:
            phase_image = image_to_phase(image_path, width, height)
            phase_stack = np.repeat(
                phase_image[np.newaxis, :, :],
                HOLOGRAMS_PER_FRAME,
                axis=0,
            )
            phase_stack = np.ascontiguousarray(phase_stack, dtype=np.float32)
            result = plm.bitpack_and_insert_gpu(phase_stack, index)

        if result == 0:
            raise RuntimeError(f"Failed to upload {image_path}")

    sequence = np.arange(sequence_slots, dtype=np.uint64) % len(image_paths)
    result = plm.set_frame_sequence(sequence)
    if result == 0:
        raise RuntimeError("Failed to set PLMCtrl frame sequence")
    plm.set_frame(0)


def run_manual_image_sequence(
    plm,
    frame_count: int,
    image_duration: float,
    cycles: int,
) -> None:
    if image_duration <= 0:
        raise ValueError("--image-duration must be greater than 0")
    if cycles < 0:
        raise ValueError("--cycles must be 0 or greater")

    cycle_text = "forever" if cycles == 0 else f"{cycles} cycle(s)"
    print(f"Repeating {frame_count} images every {image_duration:g}s, {cycle_text}.")
    print("Press Ctrl+C to stop.")

    completed_cycles = 0
    try:
        while cycles == 0 or completed_cycles < cycles:
            for frame_index in range(frame_count):
                plm.set_frame(frame_index)
                print(f"Displaying image {frame_index + 1}/{frame_count}")
                time.sleep(image_duration)
            completed_cycles += 1
    except KeyboardInterrupt:
        print("\nStopping image sequence...")


def run_plmctrl_image_sequence(
    plm,
    frame_count: int,
    sequence_slots: int,
    sequence_fps: float,
    cycles: int,
) -> None:
    if sequence_fps <= 0:
        raise ValueError("--sequence-fps must be greater than 0")
    if cycles < 0:
        raise ValueError("--cycles must be 0 or greater")

    cycle_text = "forever" if cycles == 0 else f"{cycles} cycle(s)"
    print(
        f"Running {frame_count} full-frame images at about {sequence_fps:g} fps "
        f"through {sequence_slots} PLMCtrl sequence slots, {cycle_text}."
    )
    print("No bitplane packing is used in this mode.")
    print("Press Ctrl+C to stop.")

    # plmctrl advances once per rendered frame and decrements after presentation.
    # Passing sequence_slots - 1 displays all configured order slots once.
    frames_to_display = max(1, sequence_slots - 1)
    # Restart just before the sequence goes inactive so repeated playback stays smooth.
    restart_delay = max(0.0, (sequence_slots - 0.5) / sequence_fps)

    completed_cycles = 0
    try:
        while cycles == 0 or completed_cycles < cycles:
            result = plm.start_sequence(frames_to_display)
            if result == 0:
                raise RuntimeError("Failed to start PLMCtrl frame sequence")
            time.sleep(restart_delay)
            completed_cycles += 1
    except KeyboardInterrupt:
        print("\nStopping image sequence...")


def run_packed_bitplane_display(
    connection_type: int,
    cycles: int,
) -> None:
    if cycles < 0:
        raise ValueError("--cycles must be 0 or greater")

    rgb_frame_rate = DISPLAY_FRAME_RATE[connection_type]
    bitplane_rate = rgb_frame_rate * PACKED_BITPLANE_COUNT
    print(
        f"Displaying packed RGB bitplanes at approximately {bitplane_rate:g} Hz "
        f"({rgb_frame_rate:g} Hz x {PACKED_BITPLANE_COUNT} bitplanes)."
    )
    print(
        "Packed mode uploads one RGB frame; the PLM advances the images as "
        "bitplanes, so Python will not print per-image 'Displaying image' lines."
    )

    if cycles == 0:
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping packed bitplane display...")
        return

    time.sleep(cycles / rgb_frame_rate)


def main() -> int:
    args = parse_args()

    if args.playback_mode == "frame-sequence":
        print(
            "Frame-sequence mode: uploading all selected images and looping "
            "them as separate PLM frames at the video refresh rate."
        )
    else:
        print(
            "Packed-bitplane mode: uploading one RGB frame that contains "
            "multiple images in its 24 bitplanes."
        )

    if args.images:
        image_paths = [resolve_input_path(path) for path in args.images]
    elif args.no_default_images:
        image_paths = []
    else:
        image_paths = list(DEFAULT_IMAGE_SEQUENCE)

    if args.playback_mode == "bitplane-package" and image_paths:
        if args.packed_image_count <= 0:
            raise SystemExit("--packed-image-count must be greater than 0")
        if len(image_paths) < args.packed_image_count:
            raise SystemExit(
                "--packed-image-count cannot exceed the number of available images"
            )
        image_paths = image_paths[: args.packed_image_count]
    elif args.playback_mode == "frame-sequence" and image_paths:
        if args.sequence_image_count < 0:
            raise SystemExit("--sequence-image-count must be 0 or greater")
        if args.sequence_image_count:
            if len(image_paths) < args.sequence_image_count:
                raise SystemExit(
                    "--sequence-image-count cannot exceed the number of available images"
                )
            image_paths = image_paths[: args.sequence_image_count]

    if (
        args.playback_mode == "frame-sequence"
        and image_paths
        and args.max_frames < len(image_paths)
    ):
        raise SystemExit("--max-frames must be at least the number of --images")

    with plmctrl_runtime():
        PLMController = load_plm_controller_class()

        try:
            plm = PLMController(
                args.max_frames,
                args.width,
                args.height,
                dll_path=str(DLL_PATH),
                x0=args.x0,
                y0=args.y0,
            )
        except OSError as exc:
            raise SystemExit(
                f"Could not load {DLL_PATH}.\n"
                f"Original error: {exc}\n"
                "Use 64-bit Python and keep the DLL dependencies in plmctrl-main/bin."
            ) from exc

        if args.check_only:
            print("PLMCtrl wrapper and DLL loaded successfully.")
            return 0

        if args.connection == "auto":
            connection_name = (
                "displayport"
                if args.playback_mode == "bitplane-package"
                else "hdmi"
            )
        else:
            connection_name = args.connection

        connection_type = CONNECTION_TYPES[connection_name]
        play_mode = PLAY_MODES[args.play_mode]
        port_swap = PORT_SWAPS[args.port_swap]

        use_windowed_swapchain = args.windowed or (
            args.playback_mode == "frame-sequence" and not args.exclusive_fullscreen
        )
        if use_windowed_swapchain:
            print("Using windowed DirectX swapchain for the PLM UI.")
            plm.set_windowed(True)
        else:
            print("Using exclusive/fullscreen DirectX swapchain for the PLM UI.")
            plm.set_windowed(False)

        configure_plm(plm, play_mode, connection_type, port_swap, args.settle_seconds)

        ui_started = args.start_ui or bool(image_paths)

        if ui_started:
            print("Starting PLM UI...")
            plm.start_ui()
            time.sleep(args.ui_warmup_seconds)

        if image_paths and args.playback_mode == "bitplane-package":
            upload_packed_bitplane_frame(
                plm,
                image_paths,
                args.width,
                args.height,
                args.bitplane_layout,
                args.allow_threshold_packed_images,
            )
        elif image_paths:
            upload_image_sequence(
                plm,
                image_paths,
                args.width,
                args.height,
                args.max_frames,
            )

        should_call_play = not args.no_play and not (
            image_paths
            and args.playback_mode == "frame-sequence"
            and args.advance_mode == "plmctrl"
        )

        if should_call_play:
            print("Starting PLM playback...")
            plm.play()

        if image_paths and args.playback_mode == "bitplane-package":
            try:
                run_packed_bitplane_display(connection_type, args.cycles)
            finally:
                if not args.no_play:
                    plm.stop()
                plm.stop_ui()
        elif image_paths:
            try:
                if args.advance_mode == "plmctrl":
                    sequence_fps = args.sequence_fps or DISPLAY_FRAME_RATE[
                        connection_type
                    ]
                    run_plmctrl_image_sequence(
                        plm,
                        len(image_paths),
                        args.max_frames,
                        sequence_fps,
                        args.cycles,
                    )
                else:
                    run_manual_image_sequence(
                        plm,
                        len(image_paths),
                        args.image_duration,
                        args.cycles,
                    )
            finally:
                if not args.no_play:
                    plm.stop()
                plm.stop_ui()
        elif args.start_ui:
            print("PLM UI is running. Press Ctrl+C to stop it.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping PLM UI...")
                if not args.no_play:
                    plm.stop()
                plm.stop_ui()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
