import argparse
import warnings
import numpy as np
import cv2
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
from PIL import Image, ImageOps

import torch
from torch import nn


class DeviceLibrary:
    """
    Device library trimmed to the 0.67 TI PLM only.
    """

    PHASE_LUT_REFERENCE_WAVELENGTH_M = 532e-9
    PHASE_LUT_WAVELENGTHS_M = {
        "EMPIRICAL": 532e-9,
        "EMPIRICAL_532NM": 532e-9,
        "EMPIRICAL_632.8NM": 632.8e-9,
        "EMPIRICAL_6328NM": 632.8e-9,
        "EMPIRICAL_633NM": 632.8e-9,
    }
    PHASE_LUT_SCHEME_LABELS = {
        "EMPIRICAL": "Empirical_532nm",
        "EMPIRICAL_532NM": "Empirical_532nm",
        "EMPIRICAL_632.8NM": "Empirical_632.8nm",
        "EMPIRICAL_6328NM": "Empirical_632.8nm",
        "EMPIRICAL_633NM": "Empirical_632.8nm",
    }
    DEVICE_ALIASES = {
        "0.67": None,
        "0.67_532NM": "EMPIRICAL_532NM",
        "0.67_632.8NM": "EMPIRICAL_632.8NM",
        "0.67_6328NM": "EMPIRICAL_6328NM",
        "0.67_633NM": "EMPIRICAL_633NM",
    }
    BASE_PHASE_LUT_532NM = np.array([
        0.0000,
        0.0073,
        0.0130,
        0.0285,
        0.0515,
        0.0618,
        0.1092,
        0.1793,
        0.2790,
        0.3018,
        0.3581,
        0.4289,
        0.5304,
        0.5929,
        0.7589,
        0.9375,
        1.0000
    ], dtype=np.float64)

    @classmethod
    def _phase_lut_for_wavelength(cls, wavelength_m):
        pLevel = cls.BASE_PHASE_LUT_532NM.copy()
        if not np.isclose(wavelength_m, cls.PHASE_LUT_REFERENCE_WAVELENGTH_M):
            pLevel[:-1] = np.remainder(
                pLevel[:-1] * cls.PHASE_LUT_REFERENCE_WAVELENGTH_M / wavelength_m,
                1
            )
            pLevel[-1] = 1.0
        return pLevel

    def defineDevice(self, device, discScheme="Empirical", lambdaGlobal=None, lambdaM=None):
        device = str(device).upper()
        discScheme = str(discScheme).upper()

        if device not in self.DEVICE_ALIASES:
            raise ValueError("TIPLMSuite only supports the 0.67 device.")
        alias_disc_scheme = self.DEVICE_ALIASES[device]
        if alias_disc_scheme is not None and discScheme == "EMPIRICAL":
            discScheme = alias_disc_scheme
        if discScheme not in self.PHASE_LUT_WAVELENGTHS_M:
            raise ValueError("TIPLMSuite includes empirical 0.67 phase tables for 532 nm and 632.8 nm.")

        phase_lut_wavelength_m = self.PHASE_LUT_WAVELENGTHS_M[discScheme]
        phase_lut_scheme_label = self.PHASE_LUT_SCHEME_LABELS[discScheme]
        pLevel = self._phase_lut_for_wavelength(phase_lut_wavelength_m)

        if lambdaGlobal is not None and lambdaGlobal > 0:
            reference_wavelength_m = phase_lut_wavelength_m if lambdaM is None else lambdaM
            pLevel[:-1] = np.remainder(pLevel[:-1] * lambdaGlobal / reference_wavelength_m, 1)
            pLevel[-1] = 1.0

        return {
            "device": "0.67",
            "discScheme": phase_lut_scheme_label,
            "phase_lut_wavelength_m": phase_lut_wavelength_m,
            "pitchW": 10.8e-6,
            "pitchH": 10.8e-6,
            "w": 1358,
            "h": 800,
            "nLevel": 16,
            "pLevel": pLevel,
            "memory_layout": [[1, 3], [0, 2]],
            "memory_lut": [3, 2, 1, 7, 0, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
            "evm_padding": 0,
            "evm_fliplr": True,
            "evm_flipud": False
        }

    def formatPLM(self, DeviceDictionary, CGH_state_map):
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuite only supports 0.67 PLM formatting.")
        return self.formatPLM_p67(CGH_state_map, DeviceDictionary["nLevel"])

    def formatPLM_p67(self, CGH_state_map, nLevel):
        if nLevel != 16:
            raise ValueError("0.67 PLM formatting expects 16 phase levels.")

        electrode_bits = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 1, 0, 0],
        ], dtype=np.float64)

        stateq = np.clip(CGH_state_map.astype(int), 0, nLevel - 1)
        rows, cols = stateq.shape
        pslm_logical = np.zeros((rows * 2, cols * 2), dtype=np.float64)

        pslm_logical[0::2, 1::2] = electrode_bits[stateq, 0]  # E3
        pslm_logical[1::2, 1::2] = electrode_bits[stateq, 1]  # E2
        pslm_logical[0::2, 0::2] = electrode_bits[stateq, 2]  # E1
        pslm_logical[1::2, 0::2] = electrode_bits[stateq, 3]  # E0

        return np.fliplr(pslm_logical) * 255


class CGHGenerator:
    """
    CGH generator trimmed to the 0.67 device and Fourier ADAM/ADAMwGS.
    """
    SCALE_EPS = 1e-20

    def __init__(self):
        self.CGH_mapped = None
        self.CGH_phase = None
        self.CGH_output_cont = None
        self.CGH_output_phase_disc = None
        self.CGH_output_state_disc = None
        self.CGH_output_disc = None
        self.imRecovered_cont = None
        self.imRecovered_disc = None
        self.seedPhase = None
        self.imTarget = None
        self.imTarget_orig = None
        self.deviceLibary = DeviceLibrary()
        self.lambda_m = 532e-9
        self.pitchW = None
        self.pitchH = None
        self.bitPlanes = 1
        self.show_images = False
        self.lossMode = "auto"
        self.angleMuxTargets = None
        self.angleMuxPreparedFiles = None
        self.angleMuxViewShifts = None
        self.angleMuxViewAnglesDeg = None
        self.angleMuxRecovered_cont = None
        self.angleMuxRecovered_disc = None
        self.angleMuxOrderDisplayMask = None
        self.angleMuxOrderTrainingMask = None
        self.angleMuxMetrics = None
        self.MLDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _centerCropToAspect(pil_image, target_size):
        target_w, target_h = target_size
        src_w, src_h = pil_image.size
        target_aspect = target_w / target_h
        src_aspect = src_w / src_h

        if abs(src_aspect - target_aspect) < 1e-12:
            return pil_image

        if src_aspect > target_aspect:
            crop_w = int(round(src_h * target_aspect))
            left = max(0, (src_w - crop_w) // 2)
            return pil_image.crop((left, 0, left + crop_w, src_h))

        crop_h = int(round(src_w / target_aspect))
        top = max(0, (src_h - crop_h) // 2)
        return pil_image.crop((0, top, src_w, top + crop_h))

    @staticmethod
    def _fitImageToTarget(pil_image, target_size, fitMode="stretch", background=255):
        fit_mode = str(fitMode).replace("-", "_").replace(" ", "_").lower()
        if fit_mode in ("stretch", "fill", "resize"):
            return pil_image.resize(target_size, Image.Resampling.LANCZOS)
        if fit_mode in ("crop", "center_crop", "cover"):
            return CGHGenerator._centerCropToAspect(pil_image, target_size).resize(
                target_size,
                Image.Resampling.LANCZOS,
            )
        if fit_mode in ("contain", "letterbox", "fit"):
            fitted = ImageOps.contain(pil_image, target_size, Image.Resampling.LANCZOS)
            canvas = Image.new("L", target_size, int(background))
            left = (target_size[0] - fitted.size[0]) // 2
            top = (target_size[1] - fitted.size[1]) // 2
            canvas.paste(fitted, (left, top))
            return canvas
        raise ValueError("fitMode must be 'stretch', 'crop', or 'contain'.")

    @staticmethod
    def _photoToneMap(grayscale_image, blackPercentile=1.0, whitePercentile=99.0, gamma=0.9, sharpenAmount=0.35):
        arr = np.asarray(grayscale_image, dtype=np.float32) / 255.0
        lo = float(np.percentile(arr, blackPercentile))
        hi = float(np.percentile(arr, whitePercentile))
        if hi <= lo + 1e-6:
            lo = float(arr.min())
            hi = float(arr.max())
        if hi > lo + 1e-6:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        if gamma > 0:
            arr = np.power(arr, float(gamma))
        if sharpenAmount > 0:
            blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.0, sigmaY=1.0)
            arr = np.clip(arr + float(sharpenAmount) * (arr - blur), 0.0, 1.0)
        return arr

    @staticmethod
    def _serpentineErrorDiffusion(values, threshold=0.5, method="floyd_steinberg"):
        work = np.asarray(values, dtype=np.float32).copy()
        height, width = work.shape
        out = np.zeros((height, width), dtype=np.uint8)
        method_key = str(method).replace("-", "_").replace(" ", "_").lower()

        for y in range(height):
            left_to_right = (y % 2) == 0
            x_range = range(width) if left_to_right else range(width - 1, -1, -1)
            direction = 1 if left_to_right else -1

            for x in x_range:
                old = work[y, x]
                new = 1.0 if old >= threshold else 0.0
                out[y, x] = 255 if new > 0 else 0
                err = old - new

                if method_key == "atkinson":
                    weights = [
                        (direction, 0, 1 / 8),
                        (2 * direction, 0, 1 / 8),
                        (-direction, 1, 1 / 8),
                        (0, 1, 1 / 8),
                        (direction, 1, 1 / 8),
                        (0, 2, 1 / 8),
                    ]
                elif method_key in ("jarvis", "jjn", "jarvis_judice_ninke"):
                    weights = [
                        (direction, 0, 7 / 48),
                        (2 * direction, 0, 5 / 48),
                        (-2 * direction, 1, 3 / 48),
                        (-direction, 1, 5 / 48),
                        (0, 1, 7 / 48),
                        (direction, 1, 5 / 48),
                        (2 * direction, 1, 3 / 48),
                        (-2 * direction, 2, 1 / 48),
                        (-direction, 2, 3 / 48),
                        (0, 2, 5 / 48),
                        (direction, 2, 3 / 48),
                        (2 * direction, 2, 1 / 48),
                    ]
                elif method_key in ("floyd", "floyd_steinberg", "fs"):
                    weights = [
                        (direction, 0, 7 / 16),
                        (-direction, 1, 3 / 16),
                        (0, 1, 5 / 16),
                        (direction, 1, 1 / 16),
                    ]
                else:
                    raise ValueError("ditherMethod must be 'floyd_steinberg', 'jarvis', or 'atkinson'.")

                for dx, dy, weight in weights:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        work[ny, nx] += err * weight

        return out

    @staticmethod
    def _thresholdMatrixDither(values, threshold=0.5):
        bayer8 = np.array([
            [0, 48, 12, 60, 3, 51, 15, 63],
            [32, 16, 44, 28, 35, 19, 47, 31],
            [8, 56, 4, 52, 11, 59, 7, 55],
            [40, 24, 36, 20, 43, 27, 39, 23],
            [2, 50, 14, 62, 1, 49, 13, 61],
            [34, 18, 46, 30, 33, 17, 45, 29],
            [10, 58, 6, 54, 9, 57, 5, 53],
            [42, 26, 38, 22, 41, 25, 37, 21],
        ], dtype=np.float32)
        matrix = (bayer8 + 0.5) / 64.0
        tiled = np.tile(matrix, (
            int(np.ceil(values.shape[0] / matrix.shape[0])),
            int(np.ceil(values.shape[1] / matrix.shape[1])),
        ))[:values.shape[0], :values.shape[1]]
        adjusted = np.clip(values + (threshold - 0.5), 0.0, 1.0)
        return (adjusted >= tiled).astype(np.uint8) * 255

    @staticmethod
    def _axisOrderWindow(length, fraction=1.0, featherFraction=0.0):
        fraction = float(fraction)
        featherFraction = max(0.0, float(featherFraction))
        if fraction <= 0.0:
            raise ValueError("Order window fractions must be > 0.")
        if fraction >= 1.0:
            return np.ones((length,), dtype=np.float32)

        center = (length - 1) / 2.0
        half_width = max(1.0, fraction * length / 2.0)
        feather = featherFraction * length
        distance = np.abs(np.arange(length, dtype=np.float32) - center)

        if feather <= 1e-6:
            return (distance <= half_width).astype(np.float32)

        mask = np.clip((half_width + feather - distance) / feather, 0.0, 1.0)
        mask[distance <= half_width] = 1.0
        return mask.astype(np.float32)

    @staticmethod
    def _orderWindowMasks(
        shape,
        widthFraction=1.0,
        heightFraction=1.0,
        featherFraction=0.03,
        ShiftFOV=True,
        FlipUD=False,
        FlipLR=False,
    ):
        rows, cols = shape
        x_mask = CGHGenerator._axisOrderWindow(cols, widthFraction, featherFraction)
        y_mask = CGHGenerator._axisOrderWindow(rows, heightFraction, featherFraction)
        display_mask = np.outer(y_mask, x_mask).astype(np.float32)

        training_mask = display_mask.copy()
        if ShiftFOV:
            training_mask = np.roll(
                training_mask,
                shift=(int(round(rows / 2)), int(round(cols / 2))),
                axis=(0, 1),
            )
        if FlipUD:
            training_mask = np.flipud(training_mask).copy()
        if FlipLR:
            training_mask = np.fliplr(training_mask).copy()

        return display_mask, training_mask.astype(np.float32)

    @staticmethod
    def prepareOneBitImage(
        filename,
        outputFilename=None,
        targetSize=(1358, 800),
        threshold=0.5,
        dither=True,
        ditherMethod="floyd_steinberg",
        fitMode="stretch",
        blackPercentile=1.0,
        whitePercentile=99.0,
        gamma=0.9,
        sharpenAmount=0.35,
        invert=False,
    ):
        source_path = Path(filename)
        if outputFilename is None:
            output_path = source_path.with_name(source_path.stem + "_1358x800_1bit.bmp")
        else:
            output_path = Path(outputFilename)

        with Image.open(source_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            exact_one_bit = pil_image.mode == "1" and pil_image.size == targetSize and not invert
            if exact_one_bit:
                one_bit = pil_image.copy()
            else:
                grayscale = pil_image.convert("L")
                resized = CGHGenerator._fitImageToTarget(grayscale, targetSize, fitMode=fitMode)
                if invert:
                    resized = ImageOps.invert(resized)
                tone_mapped = CGHGenerator._photoToneMap(
                    resized,
                    blackPercentile=blackPercentile,
                    whitePercentile=whitePercentile,
                    gamma=gamma,
                    sharpenAmount=sharpenAmount,
                )
                if dither:
                    dither_key = str(ditherMethod).replace("-", "_").replace(" ", "_").lower()
                    if dither_key in ("ordered", "bayer", "bayer8"):
                        one_bit_arr = CGHGenerator._thresholdMatrixDither(tone_mapped, threshold=threshold)
                    else:
                        one_bit_arr = CGHGenerator._serpentineErrorDiffusion(
                            tone_mapped,
                            threshold=threshold,
                            method=ditherMethod,
                        )
                    one_bit = Image.fromarray(one_bit_arr).convert("1")
                else:
                    one_bit_arr = (tone_mapped >= float(threshold)).astype(np.uint8) * 255
                    one_bit = Image.fromarray(one_bit_arr).convert("1")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        one_bit.save(output_path, "BMP")
        target = np.asarray(one_bit.convert("L"), dtype=np.float64) / 255.0
        return target, str(output_path)

    @staticmethod
    def _carrier(shape, shift_pixels, xp=np):
        rows, cols = shape
        shift_x, shift_y = shift_pixels
        if xp is torch:
            yy = torch.arange(rows, dtype=torch.float32)
            xx = torch.arange(cols, dtype=torch.float32)
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            phase = 2 * np.pi * (float(shift_x) * xx / cols + float(shift_y) * yy / rows)
            return torch.exp(1j * phase)

        yy, xx = np.indices((rows, cols), dtype=np.float64)
        phase = 2 * np.pi * (float(shift_x) * xx / cols + float(shift_y) * yy / rows)
        return np.exp(1j * phase)

    @staticmethod
    def _viewShiftToAngleDeg(shift_pixels, DeviceDictionary, wavelength_m):
        shift_x, shift_y = shift_pixels
        aperture_w = DeviceDictionary["w"] * DeviceDictionary["pitchW"]
        aperture_h = DeviceDictionary["h"] * DeviceDictionary["pitchH"]
        sin_x = np.clip(wavelength_m * float(shift_x) / aperture_w, -1.0, 1.0)
        sin_y = np.clip(wavelength_m * float(shift_y) / aperture_h, -1.0, 1.0)
        return (float(np.degrees(np.arcsin(sin_x))), float(np.degrees(np.arcsin(sin_y))))

    @staticmethod
    def _circularShiftSeparation(shift_a, shift_b, period):
        period = float(period)
        if period <= 0:
            return 0.0
        delta = abs(float(shift_a) - float(shift_b)) % period
        return float(min(delta, period - delta))

    @staticmethod
    def _angleMuxOrderSeparation(shape, view_shifts):
        rows, cols = shape
        return (
            CGHGenerator._circularShiftSeparation(view_shifts[0][0], view_shifts[1][0], cols),
            CGHGenerator._circularShiftSeparation(view_shifts[0][1], view_shifts[1][1], rows),
        )

    @staticmethod
    def _angleMuxWindowSupport(shape, widthFraction, heightFraction, featherFraction):
        rows, cols = shape
        feather = max(0.0, float(featherFraction))
        width_fraction = float(widthFraction)
        height_fraction = float(heightFraction)
        x_support = np.inf if width_fraction >= 0.999 else width_fraction * cols / 2.0 + feather * cols
        y_support = np.inf if height_fraction >= 0.999 else height_fraction * rows / 2.0 + feather * rows
        return float(x_support), float(y_support)

    @staticmethod
    def _warnIfAngleMuxOrdersOverlap(
        shape,
        view_shifts,
        orderWindowWidthFraction,
        orderWindowHeightFraction,
        orderWindowFeatherFraction,
    ):
        sep_x, sep_y = CGHGenerator._angleMuxOrderSeparation(shape, view_shifts)
        support_x, support_y = CGHGenerator._angleMuxWindowSupport(
            shape,
            orderWindowWidthFraction,
            orderWindowHeightFraction,
            orderWindowFeatherFraction,
        )

        if not np.isfinite(support_x) and not np.isfinite(support_y):
            warnings.warn(
                "Angle-multiplexed views are using the full replay frame, so both angular orders "
                "can be visible from each viewing angle. Use orderWindowWidthFraction < 1 or "
                "orderWindowHeightFraction < 1 to isolate the selected order.",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        # Each window has support radius `support_x`. Two windows at ±(sep_x/2) overlap
        # when support_x > sep_x/2, i.e. when sep_x < 2 * support_x.
        if sep_x < 2 * support_x and sep_y < 2 * support_y:
            warnings.warn(
                "Angle-multiplexed view carriers overlap after Fourier wrap-around "
                f"(separation x/y = {sep_x:.1f}/{sep_y:.1f} pixels, "
                f"window support x/y = {support_x:.1f}/{support_y:.1f} pixels). "
                "Reduce orderWindowWidthFraction/orderWindowHeightFraction, "
                "or use a symmetric viewShiftPixels near image_width / 4.",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _resolveAngleMuxViewShifts(shape, viewShiftPixels=330, viewShifts=None):
        _, cols = shape
        if viewShifts is None:
            requested_shift = abs(float(viewShiftPixels))
            max_symmetric_shift = max(1.0, cols / 4.0)
            shift = min(requested_shift, max_symmetric_shift)
            if requested_shift > max_symmetric_shift:
                warnings.warn(
                    f"viewShiftPixels={requested_shift:.1f} wraps the two symmetric angular "
                    f"orders closer together for a {cols}-pixel-wide hologram. Using "
                    f"{shift:.1f} pixels instead, near image_width / 4.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return [(-shift, 0.0), (shift, 0.0)]

        if len(viewShifts) != 2:
            raise ValueError("viewShifts must contain two (x, y) pixel-shift pairs.")
        return [(float(v[0]), float(v[1])) for v in viewShifts]

    def _preprocessAngleMuxTargets(
        self,
        filenames,
        DeviceDictionary,
        preparedDir=None,
        threshold=0.5,
        dither=True,
        ditherMethod="floyd_steinberg",
        fitMode="stretch",
        blackPercentile=1.0,
        whitePercentile=99.0,
        gamma=0.9,
        sharpenAmount=0.35,
        invert=False,
        ShiftFOV=True,
        FlipUD=False,
        FlipLR=False,
    ):
        targets = []
        prepared_files = []
        target_size = (DeviceDictionary["w"], DeviceDictionary["h"])
        prepared_dir = Path(preparedDir) if preparedDir is not None else None

        for filename in filenames:
            source_path = Path(filename)
            output_path = None
            if prepared_dir is not None:
                output_path = prepared_dir / (source_path.stem + "_1358x800_1bit.bmp")

            target, prepared_file = CGHGenerator.prepareOneBitImage(
                source_path,
                outputFilename=output_path,
                targetSize=target_size,
                threshold=threshold,
                dither=dither,
                ditherMethod=ditherMethod,
                fitMode=fitMode,
                blackPercentile=blackPercentile,
                whitePercentile=whitePercentile,
                gamma=gamma,
                sharpenAmount=sharpenAmount,
                invert=invert,
            )

            if ShiftFOV:
                target = np.roll(
                    target,
                    shift=(int(round(target.shape[0] / 2)), int(round(target.shape[1] / 2))),
                    axis=(0, 1),
                )
            if FlipUD:
                target = np.flipud(target).copy()
            if FlipLR:
                target = np.fliplr(target).copy()

            targets.append(target.astype(np.float32))
            prepared_files.append(prepared_file)

        return np.stack(targets, axis=0), prepared_files

    def _initialAngleMuxPhase(self, targets, view_shifts, initialPhase="BACKPROP"):
        rows, cols = targets.shape[-2:]
        initial = str(initialPhase).upper()
        hologram_field = np.zeros((rows, cols), dtype=np.complex128)

        for view_idx, target in enumerate(targets):
            if initial in ("RANDOM", "RANDOM_PHASE"):
                target_phase = np.exp(1j * 2 * np.pi * np.random.rand(rows, cols))
            else:
                target_phase = 1.0
            desired_field = np.fft.fft2(target.astype(np.complex128) * target_phase)
            carrier = CGHGenerator._carrier((rows, cols), view_shifts[view_idx], xp=np)
            hologram_field = hologram_field + desired_field * carrier

        if np.max(np.abs(hologram_field)) < 1e-20:
            return 2 * np.pi * np.random.rand(rows, cols)

        return np.mod(np.angle(hologram_field), 2 * np.pi)

    def _recoverAngleMuxViews(
        self,
        phase,
        targets,
        view_shifts,
        ShiftFOV=True,
        trainingMask=None,
        displayMask=None,
        maskRecoveredOrders=True,
    ):
        field = np.exp(1j * phase)
        views = []
        for view_idx, shift in enumerate(view_shifts):
            carrier = CGHGenerator._carrier(phase.shape, shift, xp=np)
            view_field = np.fft.ifft2(field * np.conj(carrier))
            amp = np.abs(view_field)
            target = targets[view_idx]
            mask = None
            if trainingMask is not None:
                mask = trainingMask[view_idx] if np.asarray(trainingMask).ndim == 3 else trainingMask
            if mask is None:
                scale = np.mean(amp * target) / (np.mean(amp ** 2) + 1e-10)
            else:
                scale = np.sum(amp * target * mask) / (np.sum((amp ** 2) * mask) + 1e-10)
            amp = scale * amp
            if ShiftFOV:
                amp = np.roll(
                    amp,
                    shift=(int(amp.shape[0] / 2), int(amp.shape[1] / 2)),
                    axis=(0, 1),
                )
            if maskRecoveredOrders and displayMask is not None:
                display_mask = displayMask[view_idx] if np.asarray(displayMask).ndim == 3 else displayMask
                amp = amp * display_mask
            views.append(amp)
        return np.asarray(views)

    def createAngleMultiplexedCGH(
        self,
        DeviceDictionary,
        filenames,
        outputPreparedDir=None,
        viewShiftPixels=330,
        viewShifts=None,
        numIter=800,
        learningRate=0.08,
        initialPhase="BACKPROP",
        threshold=0.5,
        dither=True,
        ditherMethod="floyd_steinberg",
        fitMode="stretch",
        blackPercentile=1.0,
        whitePercentile=99.0,
        gamma=0.9,
        sharpenAmount=0.35,
        invert=False,
        FlipUD=True,
        FlipLR=False,
        ShiftFOV=True,
        orderWindowWidthFraction=0.38,
        orderWindowHeightFraction=1.0,
        orderWindowFeatherFraction=0.03,
        outsideOrderWeight=0.25,
        maskRecoveredOrders=True,
        showImages=False,
    ):
        if len(filenames) != 2:
            raise ValueError("createAngleMultiplexedCGH expects exactly two image filenames.")
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuiteHologram only supports the 0.67 device.")
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")

        self.show_images = showImages
        self.bitPlanes = 1
        self.pitchW = DeviceDictionary["pitchW"]
        self.pitchH = DeviceDictionary["pitchH"]

        view_shifts = CGHGenerator._resolveAngleMuxViewShifts(
            (DeviceDictionary["h"], DeviceDictionary["w"]),
            viewShiftPixels=viewShiftPixels,
            viewShifts=viewShifts,
        )
        CGHGenerator._warnIfAngleMuxOrdersOverlap(
            (DeviceDictionary["h"], DeviceDictionary["w"]),
            view_shifts,
            orderWindowWidthFraction,
            orderWindowHeightFraction,
            orderWindowFeatherFraction,
        )

        targets, prepared_files = self._preprocessAngleMuxTargets(
            filenames,
            DeviceDictionary,
            preparedDir=outputPreparedDir,
            threshold=threshold,
            dither=dither,
            ditherMethod=ditherMethod,
            fitMode=fitMode,
            blackPercentile=blackPercentile,
            whitePercentile=whitePercentile,
            gamma=gamma,
            sharpenAmount=sharpenAmount,
            invert=invert,
            ShiftFOV=ShiftFOV,
            FlipUD=FlipUD,
            FlipLR=FlipLR,
        )

        self.imTarget = targets[0]
        self.imTarget_orig = targets[0]
        self.angleMuxTargets = targets.copy()
        self.angleMuxPreparedFiles = prepared_files
        self.angleMuxViewShifts = view_shifts
        self.angleMuxViewAnglesDeg = [
            CGHGenerator._viewShiftToAngleDeg(shift, DeviceDictionary, self.lambda_m)
            for shift in view_shifts
        ]
        display_order_mask, training_order_mask = CGHGenerator._orderWindowMasks(
            targets.shape[-2:],
            widthFraction=orderWindowWidthFraction,
            heightFraction=orderWindowHeightFraction,
            featherFraction=orderWindowFeatherFraction,
            ShiftFOV=ShiftFOV,
            FlipUD=FlipUD,
            FlipLR=FlipLR,
        )
        use_order_window = (
            float(orderWindowWidthFraction) < 0.999
            or float(orderWindowHeightFraction) < 0.999
        )
        if use_order_window:
            self.angleMuxOrderDisplayMask = display_order_mask
            self.angleMuxOrderTrainingMask = training_order_mask
        else:
            self.angleMuxOrderDisplayMask = None
            self.angleMuxOrderTrainingMask = None

        phase_init = self._initialAngleMuxPhase(targets, view_shifts, initialPhase=initialPhase)
        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice, requires_grad=True)
        target_t = torch.tensor(targets, dtype=torch.float32, device=self.MLDevice)
        training_mask_t = torch.tensor(training_order_mask, dtype=torch.float32, device=self.MLDevice)
        outside_mask_t = 1.0 - training_mask_t
        inside_norm = torch.clamp(training_mask_t.sum(), min=1.0)
        outside_norm_value = float(outside_mask_t.detach().cpu().sum().item())
        outside_norm = torch.clamp(outside_mask_t.sum(), min=1.0)
        carriers = torch.stack([
            CGHGenerator._carrier(targets.shape[-2:], shift, xp=torch).to(self.MLDevice)
            for shift in view_shifts
        ], dim=0)

        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=float(learningRate))
        msel = np.zeros((int(numIter), 2), dtype=np.float64)
        total_loss_trace = np.zeros((int(numIter),), dtype=np.float64)
        best_loss = float("inf")
        best_phase = None

        for i in tqdm(range(0, int(numIter))):
            optimizer.zero_grad(set_to_none=True)
            field = torch.exp(1j * phase_hp)
            loss_val = torch.tensor(0.0, dtype=torch.float32, device=self.MLDevice)

            for view_idx in range(2):
                view_field = torch.fft.ifft2(field * torch.conj(carriers[view_idx]))
                view_amp = torch.abs(view_field)
                target = target_t[view_idx]
                if use_order_window:
                    with torch.no_grad():
                        gain = ((view_amp * target * training_mask_t).sum()
                                / (((view_amp ** 2) * training_mask_t).sum() + self.SCALE_EPS))
                    view_diff = gain * view_amp - target
                    inside_loss = ((view_diff ** 2) * training_mask_t).sum() / inside_norm
                    view_loss = inside_loss
                    if float(outsideOrderWeight) > 0.0 and outside_norm_value > 0.0:
                        outside_loss = (((gain * view_amp) ** 2) * outside_mask_t).sum() / outside_norm
                        view_loss = view_loss + float(outsideOrderWeight) * outside_loss
                else:
                    with torch.no_grad():
                        gain = (view_amp * target).mean() / ((view_amp ** 2).mean() + self.SCALE_EPS)
                    inside_loss = ((gain * view_amp - target) ** 2).mean()
                    view_loss = inside_loss
                msel[i, view_idx] = float(inside_loss.detach().cpu().item())
                loss_val = loss_val + view_loss

            total_loss_trace[i] = float(loss_val.detach().cpu().item())
            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                phase_hp.remainder_(2 * np.pi)
                if total_loss_trace[i] < best_loss:
                    best_loss = total_loss_trace[i]
                    best_phase = phase_hp.detach().clone()

        if best_phase is None:
            best_phase = phase_hp.detach().clone()

        self.CGH_output_cont = np.mod(best_phase.detach().cpu().numpy(), 2 * np.pi)
        self.CGH_output_phase_disc, self.CGH_output_state_disc = self.discretePhase(
            self.CGH_output_cont,
            DeviceDictionary["nLevel"],
            DeviceDictionary["pLevel"],
        )
        self.CGH_output_phase_disc = 2 * np.pi * self.CGH_output_phase_disc
        self.CGH_output_disc = self.CGH_output_state_disc
        self.CGH_phase = self.CGH_output_phase_disc
        self.CGH_mapped = self.deviceLibary.formatPLM(DeviceDictionary, self.CGH_output_state_disc)

        self.angleMuxRecovered_cont = self._recoverAngleMuxViews(
            self.CGH_output_cont,
            targets,
            view_shifts,
            ShiftFOV=ShiftFOV,
            trainingMask=training_order_mask if use_order_window else None,
            displayMask=display_order_mask if use_order_window else None,
            maskRecoveredOrders=maskRecoveredOrders,
        )
        self.angleMuxRecovered_disc = self._recoverAngleMuxViews(
            self.CGH_output_phase_disc,
            targets,
            view_shifts,
            ShiftFOV=ShiftFOV,
            trainingMask=training_order_mask if use_order_window else None,
            displayMask=display_order_mask if use_order_window else None,
            maskRecoveredOrders=maskRecoveredOrders,
        )
        self.imRecovered_cont = self.angleMuxRecovered_cont[0]
        self.imRecovered_disc = self.angleMuxRecovered_disc[0]

        psnr = 20 * np.log10(1 / np.sqrt(np.maximum(msel, 1e-20)))
        self.angleMuxMetrics = {
            "best_total_loss": float(best_loss),
            "view_mse": [float(np.min(msel[:, 0])), float(np.min(msel[:, 1]))],
            "view_psnr": [float(np.max(psnr[:, 0])), float(np.max(psnr[:, 1]))],
            "view_shifts_pixels": view_shifts,
            "view_angles_degrees": self.angleMuxViewAnglesDeg,
            "view_order_separation_pixels": CGHGenerator._angleMuxOrderSeparation(targets.shape[-2:], view_shifts),
            "order_window_width_fraction": float(orderWindowWidthFraction),
            "order_window_height_fraction": float(orderWindowHeightFraction),
            "outside_order_weight": float(outsideOrderWeight),
            "prepared_files": prepared_files,
            "numIter": int(numIter),
            "learningRate": float(learningRate),
        }

        print("Angle multiplexed CGH complete.")
        print("View 1 best MSE/PSNR: " + str(self.angleMuxMetrics["view_mse"][0]) + " / " + str(self.angleMuxMetrics["view_psnr"][0]))
        print("View 2 best MSE/PSNR: " + str(self.angleMuxMetrics["view_mse"][1]) + " / " + str(self.angleMuxMetrics["view_psnr"][1]))
        print("View angles in degrees (x, y): " + str(self.angleMuxMetrics["view_angles_degrees"]))

    def loadTarget(self, filename="", colorChannel=0):
        if filename != "":
            fn = filename
        else:
            Tk().withdraw()
            fn = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])

        grayscale_keys = {"gray", "grey", "l", "luma", "luminance"}
        use_grayscale = colorChannel is None or (
            isinstance(colorChannel, str) and colorChannel.lower() in grayscale_keys
        )
        if not use_grayscale:
            if isinstance(colorChannel, str):
                raise ValueError("colorChannel must be 0, 1, 2, or 'gray'.")
            if colorChannel < 0 or colorChannel > 2:
                raise ValueError("colorChannel must be 0, 1, 2, or 'gray'.")

        try:
            with Image.open(fn) as pil_image:
                pil_image = ImageOps.exif_transpose(pil_image)
                if use_grayscale or pil_image.mode in ("1", "L", "I", "I;16", "F"):
                    image = np.asarray(pil_image.convert("L"), dtype=np.float64) / 255.0
                    self.imTarget_orig = image
                    self.imTarget = self.imTarget_orig
                    return

                image = np.asarray(pil_image.convert("RGB"), dtype=np.float64) / 255.0
        except Exception as exc:
            raise FileNotFoundError(f"Could not load image: {fn}") from exc

        self.imTarget_orig = image[:, :, colorChannel]
        self.imTarget = self.imTarget_orig

    def resizeTarget(self, lambda_m, h, w, pitchW, pitchH, ShiftFOV=True, preserveAspect=True):
        image = self.imTarget
        source_shape = np.shape(image)

        if preserveAspect:
            fov_aspect_ratio = np.arcsin(lambda_m / pitchW) / np.arcsin(lambda_m / pitchH)

            if source_shape[1] / source_shape[0] < fov_aspect_ratio:
                delta = int(abs(source_shape[0] - np.round(source_shape[1] * fov_aspect_ratio)))
                if np.mod(delta, 2) == 0:
                    image = np.pad(image, ((0, 0), (delta // 2, delta // 2)), "constant")
                else:
                    image = np.pad(image, ((0, 0), (int(float(delta) // 2 + 1 / 2) + 1, int(float(delta) // 2 - 1 / 2) + 1)), "constant")
            elif source_shape[1] / source_shape[0] > fov_aspect_ratio:
                delta = int(abs(source_shape[0] - np.round(source_shape[1] / fov_aspect_ratio)))
                if np.mod(delta, 2) == 0:
                    image = np.pad(image, ((delta // 2, delta // 2), (0, 0)), "constant")
                else:
                    image = np.pad(image, ((int(float(delta) // 2 + 1 / 2) + 1, int(float(delta) // 2 - 1 / 2) + 1), (0, 0)), "constant")

        if ShiftFOV:
            image = np.roll(image, shift=(int(np.round(np.shape(image)[0] / 2)), int(np.round(np.shape(image)[1] / 2))), axis=(0, 1))

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        den = np.max(image) - np.min(image)
        if den < 1e-12:
            image = np.zeros_like(image)
        else:
            image = (image - np.min(image)) / den
        self.imTarget = image

    def binarizeTarget(self, threshold=0.5):
        if self.imTarget is None:
            return
        self.imTarget = (self.imTarget >= threshold).astype(np.float64)

    def _targetIsBinary(self):
        if self.imTarget is None:
            return False
        return np.all((self.imTarget == 0) | (self.imTarget == 1))

    def _resolveLossMode(self, lossMode):
        mode = str(lossMode).replace("-", "_").replace(" ", "_").lower()
        if mode == "auto":
            return "balanced" if self._targetIsBinary() else "mse"
        if mode not in ("mse", "balanced"):
            raise ValueError("lossMode must be 'auto', 'mse', or 'balanced'.")
        return mode

    @staticmethod
    def _torchLossWeights(target, lossMode, eps=1e-8):
        if lossMode != "balanced":
            return None

        foreground = target > 0.5
        background = ~foreground
        fg_fraction = torch.clamp(foreground.float().mean(), min=eps)
        bg_fraction = torch.clamp(background.float().mean(), min=eps)

        if bool((foreground.any() & background.any()).detach().cpu()):
            weights = torch.where(foreground, 0.5 / fg_fraction, 0.5 / bg_fraction)
            return weights / torch.clamp(weights.mean(), min=eps)

        return None

    @staticmethod
    def _torchWeightedGain(prediction, target, weights=None, eps=1e-20):
        if weights is None:
            denom = torch.clamp((prediction ** 2).mean(dim=(-1, -2), keepdim=True), min=eps)
            return (prediction * target).mean(dim=(-1, -2), keepdim=True) / denom

        denom = torch.clamp((weights * prediction ** 2).mean(dim=(-1, -2), keepdim=True), min=eps)
        return (weights * prediction * target).mean(dim=(-1, -2), keepdim=True) / denom

    @staticmethod
    def _torchWeightedMSE(prediction, target, weights=None):
        error = (prediction - target) ** 2
        if weights is not None:
            error = weights * error
        return error.mean()

    def updateTarget(self, FlipUD=False, FlipLR=False, invertTarget=False):
        if FlipUD:
            self.imTarget = np.flipud(self.imTarget).copy()
        if FlipLR:
            self.imTarget = np.fliplr(self.imTarget).copy()
        if invertTarget:
            self.imTarget = 1 - self.imTarget.copy()

    def createCGH(self, DeviceDictionary,
                  filename="", bitPlanes=1, colorChannel=0,
                  FlipUD=False, FlipLR=False, InvertTarget=False,
                  alg="ADAMWGS", numIter=1, initialPhase="Random", propMethod="Fourier",
                  ShiftFOV=True, showImages=False, binarizeTarget=False, targetThreshold=0.5,
                  preserveAspect=True,
                  lossMode="auto"):
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuite only supports the 0.67 device.")
        alg_key = str(alg).upper()
        if alg_key not in ("ADAM", "ADAMWGS"):
            raise ValueError("TIPLMSuite only supports alg='ADAM' or alg='ADAMWGS'.")
        if bitPlanes != 1:
            raise ValueError("TIPLMSuite only supports bitPlanes=1.")
        if str(propMethod).upper() != "FOURIER":
            raise ValueError("TIPLMSuite only supports propMethod='Fourier'.")

        self.show_images = showImages
        self.bitPlanes = bitPlanes
        self.loadTarget(filename, colorChannel)
        self.resizeTarget(
            self.lambda_m,
            DeviceDictionary["h"],
            DeviceDictionary["w"],
            DeviceDictionary["pitchW"],
            DeviceDictionary["pitchH"],
            ShiftFOV,
            preserveAspect,
        )
        self.pitchW = DeviceDictionary["pitchW"]
        self.pitchH = DeviceDictionary["pitchH"]
        if binarizeTarget:
            self.binarizeTarget(targetThreshold)
        self.updateTarget(FlipUD, FlipLR, InvertTarget)
        self.lossMode = self._resolveLossMode(lossMode)

        if alg_key == "ADAM":
            self.runADAM(initialPhase, numIter, propMethod, self.lossMode)
        else:
            self.runADAMwGS(DeviceDictionary["nLevel"], DeviceDictionary["pLevel"], initialPhase, numIter, propMethod, self.lossMode)

        self.CGH_output_cont = np.mod(self.CGH_output_cont, 2 * np.pi)
        self.CGH_output_phase_disc, self.CGH_output_state_disc = self.discretePhase(
            self.CGH_output_cont,
            DeviceDictionary["nLevel"],
            DeviceDictionary["pLevel"]
        )
        self.CGH_output_phase_disc = 2 * np.pi * self.CGH_output_phase_disc
        self.CGH_output_disc = self.CGH_output_state_disc

        self.recoverImg(ShiftFOV, propMethod)
        self.CGH_mapped = self.deviceLibary.formatPLM(DeviceDictionary, self.CGH_output_state_disc)
        self.CGH_phase = self.CGH_output_phase_disc

    def recoverImg(self, ShiftFOV=True, propMethod="FOURIER"):
        if self.imTarget is None:
            return

        I_cont, _ = self.reconstruct(self.CGH_output_cont, propMethod)
        I_disc, _ = self.reconstruct(self.CGH_output_phase_disc, propMethod)

        if ShiftFOV:
            shift = (int(self.imTarget.shape[0] / 2), int(self.imTarget.shape[1] / 2))
            I_cont = np.roll(I_cont, shift=shift, axis=(0, 1))
            I_disc = np.roll(I_disc, shift=shift, axis=(0, 1))

        self.imRecovered_cont = cv2.resize(I_cont, np.flip(self.imTarget.shape), interpolation=cv2.INTER_CUBIC)
        self.imRecovered_disc = cv2.resize(I_disc, np.flip(self.imTarget.shape), interpolation=cv2.INTER_CUBIC)

    def runADAM(self, initialPhase="Random", numIter=1, propMethod="Fourier", lossMode="mse"):
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")

        if initialPhase == "Random":
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
        elif initialPhase == "Unity":
            phase_init = 2 * np.pi * np.ones(self.imTarget.shape)
        elif initialPhase == "Custom":
            if self.seedPhase is None:
                print("Error: No seed phase defined. Using random seed")
                phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
            elif self.seedPhase.shape != self.imTarget.shape:
                raise ValueError("Seed phase must be the same size as target image.")
            else:
                phase_init = self.seedPhase
        else:
            print("Error: Invalid initial phase seed. Using random seed")
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])

        msel = np.zeros((numIter, 1))
        psnrl = np.zeros((numIter, 1))

        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice)
        phase_hp.requires_grad_(True)

        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=0.32)
        target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
        loss_weights = self._torchLossWeights(target_intensity, lossMode)

        best_loss = 1e10
        best_phase = None

        for i in tqdm(range(0, numIter)):
            optimizer.zero_grad()

            E_ip_inp = torch.exp(1j * phase_hp)
            E_hp = self.torchProp("forward", E_ip_inp, propMethod)
            E_hp_intensity = torch.abs(E_hp) ** 2

            with torch.no_grad():
                s = self._torchWeightedGain(E_hp_intensity, target_intensity, loss_weights, self.SCALE_EPS)

            loss_val = self._torchWeightedMSE(s * E_hp_intensity, target_intensity, loss_weights)
            msel[i] = loss_val.item()
            psnrl[i] = 20 * np.log10(1 / np.sqrt(msel[i] + 1e-20))

            if self.show_images:
                image_recon = (s * E_hp_intensity).detach().cpu().numpy()
                cv2.imshow("ADAM Image", image_recon)
                cv2.setWindowTitle("ADAM Image", "Iteration: " + str(i + 1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                cv2.waitKey(0)

            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                if loss_val.item() < best_loss:
                    best_loss = loss_val.item()
                    best_phase = phase_hp.detach().clone()

        metric_name = "Balanced MSEL" if lossMode == "balanced" else "MSEL"
        psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
        print("Best " + metric_name + ": " + str(np.min(msel)))
        print("Best " + psnr_name + ": " + str(np.max(psnrl)))

        if best_phase is None:
            best_phase = phase_hp.detach().clone()

        self.CGH_output_cont = np.mod(best_phase.detach().cpu().numpy(), 2 * np.pi)

    def runADAMwGS(self, nLevel, pLevel, initialPhase="Random", numIter=1, propMethod="Fourier", lossMode="mse"):
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")
        if np.asarray(pLevel).ndim != 1:
            raise ValueError("TIPLMSuite only supports the 1D 0.67 phase table.")

        if initialPhase == "Random":
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
        elif initialPhase == "Unity":
            phase_init = 2 * np.pi * np.ones(self.imTarget.shape)
        elif initialPhase == "Custom":
            if self.seedPhase is None:
                print("Error: No seed phase defined. Using random seed")
                phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
            elif self.seedPhase.shape != self.imTarget.shape:
                raise ValueError("Seed phase must be the same size as target image.")
            else:
                phase_init = self.seedPhase
        else:
            print("Error: Invalid initial phase seed. Using random seed")
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])

        msel = np.zeros((numIter, 1))
        deterministic_msel = np.zeros((numIter, 1))
        psnrl = np.zeros((numIter, 1))
        deterministic_psnrl = np.zeros((numIter, 1))

        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice)
        phase_hp.requires_grad_(True)

        levels = np.asarray(pLevel[:nLevel]).reshape(1, 1, -1)
        logits = np.tile(levels, (self.imTarget.shape[0], self.imTarget.shape[1], 1))
        logits = torch.tensor(logits, device=self.MLDevice, dtype=torch.float32)

        tauInitial = torch.tensor(6.5, dtype=torch.float32, device=self.MLDevice)
        tauMin = torch.tensor(3.1, dtype=torch.float32, device=self.MLDevice)
        quantizeMethod = self.torchquantize(lut=logits, levels=nLevel, tau=6.5, quantizeMethod="gumbelsoftmax")
        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=0.17)
        
        target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
        loss_weights = self._torchLossWeights(target_intensity, lossMode)

        best_loss = 1e10
        best_quantized_phase = None

        for i in tqdm(range(0, numIter)):
            anneal_progress = 1.0 if numIter <= 1 else i / (numIter - 1)
            annealRate = torch.tensor(anneal_progress, dtype=torch.float32, device=self.MLDevice)
            quantizeMethod.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)

            optimizer.zero_grad()
            quantized_phase = quantizeMethod(phase_hp)
            E_ip_inp = torch.exp(1j * quantized_phase)
            E_hp = self.torchProp("forward", E_ip_inp, propMethod)

            #E_hp_amp = torch.abs(E_hp)
            intensity = torch.abs(E_hp) ** 2

            with torch.no_grad():
                s = self._torchWeightedGain(intensity, target_intensity, loss_weights, self.SCALE_EPS)

            loss_val = self._torchWeightedMSE(s * intensity, target_intensity, loss_weights)
            msel[i] = loss_val.item()
            psnrl[i] = 20 * np.log10(1 / np.sqrt(msel[i] + 1e-20))

            with torch.no_grad():
                deterministic_quantized_phase = quantizeMethod.deterministic(phase_hp)
                deterministic_E_ip_inp = torch.exp(1j * deterministic_quantized_phase)
                deterministic_E_out = self.torchProp("forward", deterministic_E_ip_inp, propMethod)
                deterministic_intensity = torch.abs(deterministic_E_out) ** 2
                deterministic_gain = self._torchWeightedGain(deterministic_intensity, target_intensity, loss_weights, self.SCALE_EPS)
                deterministic_loss_val = self._torchWeightedMSE(deterministic_gain * deterministic_intensity, target_intensity, loss_weights)
                deterministic_msel[i] = deterministic_loss_val.item()
                deterministic_psnrl[i] = 20 * np.log10(1 / np.sqrt(deterministic_msel[i] + 1e-20))

                if deterministic_loss_val.item() < best_loss:
                    best_loss = deterministic_loss_val.item()
                    best_quantized_phase = deterministic_quantized_phase.detach().clone()

            if self.show_images:
                image_recon = self.reconstruct(deterministic_quantized_phase.cpu().detach().numpy(), propMethod)[0]
                cv2.imshow("ADAMwGS Image", image_recon)
                cv2.setWindowTitle("ADAMwGS Image", "Iteration: " + str(i + 1) + " MSE: " + str(deterministic_msel[i]) + " PSNR: " + str(deterministic_psnrl[i]))
                cv2.waitKey(0)

            loss_val.backward()
            optimizer.step()

        metric_name = "Balanced MSEL" if lossMode == "balanced" else "MSEL"
        psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
        print("Best " + metric_name + ": " + str(np.min(deterministic_msel)))
        print("Best " + psnr_name + ": " + str(np.max(deterministic_psnrl)))
        print("Best Stochastic Training " + metric_name + ": " + str(np.min(msel)))
        print("Best Stochastic Training " + psnr_name + ": " + str(np.max(psnrl)))

        if best_quantized_phase is None:
            best_quantized_phase = quantizeMethod.deterministic(phase_hp).detach().clone()

        self.CGH_output_cont = np.mod(best_quantized_phase.detach().cpu().numpy(), 2 * np.pi)

    def prop(self, direction, field_in, algorithm="FOURIER"):
        direction = str(direction).upper()
        algorithm = str(algorithm).upper()

        if direction not in ("FORWARD", "BACKWARD"):
            raise ValueError("Propagation direction must be 'forward' or 'backward'.")
        if algorithm != "FOURIER":
            raise ValueError("TIPLMSuite only supports Fourier propagation.")

        if direction == "FORWARD":
            return np.fft.ifft2(field_in)
        return np.fft.fft2(field_in)

    def torchProp(self, direction, field_in, algorithm="FOURIER"):
        direction = str(direction).upper()
        algorithm = str(algorithm).upper()

        if direction not in ("FORWARD", "BACKWARD"):
            raise ValueError("Propagation direction must be 'forward' or 'backward'.")
        if algorithm != "FOURIER":
            raise ValueError("TIPLMSuite only supports Fourier propagation.")

        if direction == "FORWARD":
            return torch.fft.ifft2(field_in)
        return torch.fft.fft2(field_in)

    @staticmethod
    def score_phase(phase, lut, s=5.0, func="prelu"):
        func_key = str(func).replace("-", "_").replace(" ", "_").lower()
        wrapped_phase = ((phase + np.pi) % (2 * np.pi)) - np.pi
        wrapped_phase = wrapped_phase.unsqueeze(-1).repeat(1, 1, lut.shape[2])
        diff = wrapped_phase - 2 * np.pi * lut
        diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
        diff /= np.pi

        if func_key == "sigmoid":
            z = s * diff
            scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
        elif func_key == "prelu" or func_key == "leaky_relu" or func_key == "leakyrelu":
            margin = 1 - diff.abs() * s
            scores = torch.where(margin < 0, margin, 0.100 * margin)
        elif func_key == "relu":
            scores = torch.relu(1 - diff.abs() * s)
        else:
            raise ValueError(f"Unknown scoring function: {func}")

        return scores * 350 * s

    class GumbelSoftmaxQuantization(nn.Module):
        def __init__(self, lut=None, num_levels=16, tau=1.0, hard=True, score_func="prelu"):
            super(CGHGenerator.GumbelSoftmaxQuantization, self).__init__()
            self.num_levels = num_levels
            self.tau = tau
            self.tauInit = tau
            self.hard = hard
            self.score_func = score_func
            if lut is not None:
                self.quant_levels = lut
            else:
                self.quant_levels = nn.Parameter(torch.linspace(0, 1, num_levels), requires_grad=False)

        def anneal_temperature(self, annealRate, tauInitial, tauMin):
            self.tau = tauInitial * torch.exp(-torch.log(tauInitial / tauMin) * annealRate)

        def forward(self, phase):
            scores = CGHGenerator.score_phase(phase, self.quant_levels, s=self.tauInit / self.tau, func=self.score_func)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.quant_levels) + 1e-20) + 1e-20)
            up_scores = (scores + gumbel_noise) / self.tau
            up_scores = up_scores - torch.max(up_scores, dim=2, keepdim=True)[0]
            soft_assignments = torch.nn.functional.softmax(up_scores, dim=2)

            if self.hard:
                indices = torch.argmax(soft_assignments, dim=2, keepdim=True)
                hard_assignments = torch.zeros_like(soft_assignments).scatter_(2, indices, 1.0)
                probs = hard_assignments.detach() + soft_assignments - soft_assignments.detach()
            else:
                probs = soft_assignments

            return torch.sum(2 * np.pi * probs * self.quant_levels, dim=2)

        def deterministic(self, phase):
            scores = CGHGenerator.score_phase(phase, self.quant_levels, s=self.tauInit / self.tau, func=self.score_func)
            indices = torch.argmax(scores, dim=2, keepdim=True)
            hard_assignments = torch.zeros_like(scores).scatter_(2, indices, 1.0)
            return torch.sum(2 * np.pi * hard_assignments * self.quant_levels, dim=2)

    def torchquantize(self, lut=None, levels=None, quantizeMethod="GUMBELSOFTMAX", tau=1.0):
        if str(quantizeMethod).upper() != "GUMBELSOFTMAX":
            raise ValueError("TIPLMSuite only supports Gumbel-softmax quantization.")
        return CGHGenerator.GumbelSoftmaxQuantization(lut=lut, num_levels=levels, tau=tau, hard=True, score_func="relu").to(self.MLDevice)

    def reconstruct(self, phase, propMethod="FOURIER"):
        E_hp = np.ones(phase.shape) * np.exp(1j * phase)
        E_ip = self.prop("forward", E_hp, propMethod)
        I = np.real(np.abs(E_ip) ** 2)

        if self.imTarget is not None:
            scale = np.mean(I * self.imTarget) / (np.mean(I ** 2) + 1e-10)
            I = scale * I

        return I, E_ip

    def discretePhase(self, phase, nLevel, pLevel):
        phase_level = np.sort(np.asarray(pLevel), axis=0)
        phaseq = np.zeros(np.shape(phase))
        stateq = np.zeros(np.shape(phase))
        phase = np.mod(phase, 2 * np.pi) / (2 * np.pi)

        partition = np.zeros(nLevel)
        for i in range(0, len(partition)):
            partition[i] = phase_level[i] + (phase_level[i + 1] - phase_level[i]) / 2
        codebook = phase_level

        for i in range(0, np.size(phase, 1)):
            stateq[:, i], phaseq[:, i] = CGHGenerator.quantiz(phase[:, i], partition, codebook)

        phaseq[phaseq == 1] = 0
        stateq[stateq == nLevel] = 0
        return phaseq, stateq

    def writeCGHToFile(self, filename, binary=False):
        if self.CGH_mapped is None:
            raise ValueError("No CGH has been generated.")

        outputCGH = cv2.normalize(self.CGH_mapped, None, 0, 255, cv2.NORM_MINMAX)
        _, binaryCGH = cv2.threshold(outputCGH, 127, 255, cv2.THRESH_BINARY)
        binaryCGH = binaryCGH.astype(np.uint8)

        if not binary:
            cv2.imwrite(filename, binaryCGH, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        else:
            pilImage = Image.fromarray(binaryCGH)
            pilImage = pilImage.convert("1")
            pilImage.save(filename, "BMP")

    def writeVarToFile(self, filename, Var):
        outputCGH = cv2.normalize(Var, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(filename, outputCGH.astype(np.uint8))

    @staticmethod
    def quantiz(sig, partition, codebook):
        index = []
        quants = []
        for s in sig:
            idx = 0
            while idx < len(partition) and s > partition[idx]:
                idx += 1
            index.append(idx)
            quants.append(codebook[idx])
        return np.array(index), np.array(quants)

    @staticmethod
    def normalize(x):
        den = np.max(x) - np.min(x)
        if den < 1e-12:
            return np.zeros_like(x)
        return (x - np.min(x)) / den


def _parse_view_shift_pair(value):
    parts = str(value).split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("View shifts must be formatted as x,y.")
    return (float(parts[0]), float(parts[1]))


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-view angle-multiplexed 0.67 PLM hologram. "
            "Each input is resized to 1358x800, saved as a photo-halftoned 1-bit BMP, "
            "and encoded into a different angular carrier."
        )
    )
    parser.add_argument("image1", help="First source image.")
    parser.add_argument("image2", help="Second source image.")
    parser.add_argument("--output", default="two_view_angle_mux_CGH.bmp", help="Mapped PLM output BMP.")
    parser.add_argument("--prepared-dir", default="angle_mux_prepared", help="Directory for 1358x800 1-bit input BMPs.")
    parser.add_argument("--view-shift-pixels", type=float, default=330.0, help="Symmetric horizontal carrier shift in FFT pixels. Values above image_width / 4 are clamped to avoid Fourier wrap-around overlap.")
    parser.add_argument("--view1-shift", type=_parse_view_shift_pair, default=None, help="Explicit first view carrier shift as x,y pixels.")
    parser.add_argument("--view2-shift", type=_parse_view_shift_pair, default=None, help="Explicit second view carrier shift as x,y pixels.")
    parser.add_argument("--num-iter", type=int, default=800, help="Adam iterations for the shared phase hologram.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Adam learning rate.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used when --no-dither is set.")
    parser.add_argument("--fit-mode", default="stretch", choices=["stretch", "crop", "contain"], help="How to fit each source image into 1358x800.")
    parser.add_argument("--dither-method", default="floyd_steinberg", choices=["floyd_steinberg", "jarvis", "atkinson", "ordered"], help="1-bit photo halftone method.")
    parser.add_argument("--black-percentile", type=float, default=1.0, help="Low percentile mapped to black before dithering.")
    parser.add_argument("--white-percentile", type=float, default=99.0, help="High percentile mapped to white before dithering.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Tone gamma before dithering. Lower brightens midtones.")
    parser.add_argument("--sharpen-amount", type=float, default=0.35, help="Unsharp mask amount before dithering.")
    parser.add_argument("--order-window-width-fraction", type=float, default=0.38, help="Central replay-order width to keep. Use < 1 to hide/suppress left and right orders. Must satisfy: widthFraction < viewShiftPixels / (image_width / 2) to avoid cross-talk between views.")
    parser.add_argument("--order-window-height-fraction", type=float, default=1.0, help="Central replay-order height to keep.")
    parser.add_argument("--order-window-feather-fraction", type=float, default=0.03, help="Soft edge size for the replay-order window.")
    parser.add_argument("--outside-order-weight", type=float, default=0.25, help="Penalty for light outside the selected replay order.")
    parser.add_argument("--show-all-orders", action="store_true", help="Display full decoded replay planes instead of masking side orders.")
    parser.add_argument("--no-dither", action="store_true", help="Use hard thresholding instead of photo dithering.")
    parser.add_argument("--invert", action="store_true", help="Invert inputs before making 1-bit BMPs.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible optimization.")
    parser.add_argument("--no-flip-ud", action="store_true", help="Disable the 0.67 EVM vertical target flip.")
    parser.add_argument("--no-shift-fov", action="store_true", help="Disable the half-frame Fourier-domain target shift.")
    parser.add_argument("--binary-output", action="store_true", help="Write mapped PLM output as a 1-bit BMP.")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if (args.view1_shift is None) != (args.view2_shift is None):
        raise ValueError("Provide both --view1-shift and --view2-shift, or neither.")
    view_shifts = None
    if args.view1_shift is not None:
        view_shifts = [args.view1_shift, args.view2_shift]

    generator = CGHGenerator()
    device_library = DeviceLibrary()
    device = device_library.defineDevice("0.67")
    generator.createAngleMultiplexedCGH(
        device,
        [args.image1, args.image2],
        outputPreparedDir=args.prepared_dir,
        viewShiftPixels=args.view_shift_pixels,
        viewShifts=view_shifts,
        numIter=args.num_iter,
        learningRate=args.learning_rate,
        threshold=args.threshold,
        dither=not args.no_dither,
        ditherMethod=args.dither_method,
        fitMode=args.fit_mode,
        blackPercentile=args.black_percentile,
        whitePercentile=args.white_percentile,
        gamma=args.gamma,
        sharpenAmount=args.sharpen_amount,
        invert=args.invert,
        FlipUD=not args.no_flip_ud,
        ShiftFOV=not args.no_shift_fov,
        orderWindowWidthFraction=args.order_window_width_fraction,
        orderWindowHeightFraction=args.order_window_height_fraction,
        orderWindowFeatherFraction=args.order_window_feather_fraction,
        outsideOrderWeight=args.outside_order_weight,
        maskRecoveredOrders=not args.show_all_orders,
        showImages=False,
    )
    generator.writeCGHToFile(args.output, binary=args.binary_output)
    print("Prepared inputs:")
    for prepared_file in generator.angleMuxPreparedFiles:
        print("  " + prepared_file)
    print("Wrote mapped CGH: " + args.output)


if __name__ == "__main__":
    main()
