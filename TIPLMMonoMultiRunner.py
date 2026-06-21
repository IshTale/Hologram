from TIPLMMonoMulti import CGHGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

try:
        import torch
except ImportError:
        torch = None


def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)


def save_normalized(filename, image):
        normalized = display_normalize(image)
        cv2.imwrite(str(filename), np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8))


def prepare_density_target(filename, outputFilename, blurSigma=1.5):
        image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        if(image is None):
                raise FileNotFoundError("Could not read image file: " + str(filename))

        image = image.astype(np.float32) / 255.0
        if(float(blurSigma) > 0):
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=float(blurSigma), sigmaY=float(blurSigma))

        image_min = float(np.min(image))
        image_max = float(np.max(image))
        if(image_max > image_min):
                image = (image - image_min) / (image_max - image_min)

        cv2.imwrite(str(outputFilename), np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8))
        return str(outputFilename)


def shift_for_display(image):
        return np.roll(
                image,
                shift=(int(image.shape[0] / 2), int(image.shape[1] / 2)),
                axis=(0, 1),
        )


def metric_weights(target, mode="balanced", eps=1e-8):
        metric_mode = str(mode).replace("-", "_").replace(" ", "_").lower()
        if(metric_mode != "balanced"):
                return None

        foreground = target > 0.5
        background = ~foreground
        if(not (np.any(foreground) and np.any(background))):
                return None

        fg_fraction = max(float(np.mean(foreground)), eps)
        bg_fraction = max(float(np.mean(background)), eps)
        weights = np.where(foreground, 0.5 / fg_fraction, 0.5 / bg_fraction)
        return weights / max(float(np.mean(weights)), eps)


def scale_to_target(intensity, target, mode="balanced", eps=1e-10):
        weights = metric_weights(target, mode=mode)
        if(weights is None):
                _, scaled = CGHGenerator.scaleIntensityToTarget(intensity, target)
                return scaled

        scale = float(np.mean(weights * intensity * target) / (np.mean(weights * intensity ** 2) + eps))
        return scale * intensity


def compute_metric(image, target, mode="balanced"):
        weights = metric_weights(target, mode=mode)
        error = (image - target) ** 2
        if(weights is not None):
                error = weights * error
        mse = float(np.mean(error))
        return mse, CGHGenerator.computePSNR(mse)


def compute_temporal_metrics(generator, shiftFOV=True, metricMode="balanced"):
        if generator.CGH_effective_field_disc_subframes is not None:
                field_stack = np.asarray(generator.CGH_effective_field_disc_subframes)
                raw_frames = []
                for frame_idx in range(field_stack.shape[0]):
                        field_ip = generator.prop("forward", field_stack[frame_idx], "FOURIER")
                        raw_frames.append(CGHGenerator.fieldToIntensity(field_ip))
                raw_frames = np.asarray(raw_frames, dtype=np.float64)
        elif generator.CGH_output_phase_disc_subframes is not None:
                phase_stack = np.asarray(generator.CGH_output_phase_disc_subframes)
                raw_frames = []
                for frame_idx in range(phase_stack.shape[0]):
                        field_hp = np.exp(1j * phase_stack[frame_idx])
                        field_ip = generator.prop("forward", field_hp, "FOURIER")
                        raw_frames.append(CGHGenerator.fieldToIntensity(field_ip))
                raw_frames = np.asarray(raw_frames, dtype=np.float64)
        else:
                raise ValueError("No temporal frame stack was generated.")

        target = np.asarray(generator.getReconstructionTarget(raw_frames.shape[-2:]), dtype=np.float64)
        target_display = shift_for_display(target) if shiftFOV else target.copy()

        frame_psnr = []
        frame_mse = []
        frame_recons_display = []
        for frame_idx in range(raw_frames.shape[0]):
                frame_recon = scale_to_target(raw_frames[frame_idx], target, mode=metricMode)
                frame_recon_display = shift_for_display(frame_recon) if shiftFOV else frame_recon
                mse, psnr = compute_metric(frame_recon_display, target_display, mode=metricMode)
                frame_mse.append(mse)
                frame_psnr.append(psnr)
                frame_recons_display.append(frame_recon_display)

        integrated_raw = np.sum(raw_frames, axis=0)
        integrated_recon = scale_to_target(integrated_raw, target, mode=metricMode)
        integrated_display = shift_for_display(integrated_recon) if shiftFOV else integrated_recon
        integrated_mse, integrated_psnr = compute_metric(integrated_display, target_display, mode=metricMode)
        standard_integrated_recon = scale_to_target(integrated_raw, target, mode="mse")
        standard_integrated_display = shift_for_display(standard_integrated_recon) if shiftFOV else standard_integrated_recon
        standard_integrated_mse, standard_integrated_psnr = compute_metric(standard_integrated_display, target_display, mode="mse")

        return {
                "metric_mode": metricMode,
                "frame_mse": frame_mse,
                "frame_psnr": frame_psnr,
                "integrated_mse": integrated_mse,
                "integrated_psnr": integrated_psnr,
                "standard_integrated_mse": standard_integrated_mse,
                "standard_integrated_psnr": standard_integrated_psnr,
                "integrated_reconstruction": integrated_display,
                "target_display": target_display,
                "frame_reconstructions": np.asarray(frame_recons_display, dtype=np.float64),
        }


def compute_single_frame_metrics(generator, shiftFOV=True, metricMode="balanced"):
        if(generator.CGH_effective_field_disc is not None):
                field_ip = generator.prop("forward", generator.CGH_effective_field_disc, "FOURIER")
                raw = CGHGenerator.fieldToIntensity(field_ip)
        else:
                field_ip = generator.prop("forward", np.exp(1j * generator.CGH_output_phase_disc), "FOURIER")
                raw = CGHGenerator.fieldToIntensity(field_ip)

        target_raw = np.asarray(generator.getReconstructionTarget(raw.shape), dtype=np.float64)
        target = shift_for_display(target_raw) if shiftFOV else target_raw.copy()

        recon = scale_to_target(raw, target_raw, mode=metricMode)
        if(shiftFOV):
                recon = shift_for_display(recon)

        mse, psnr = compute_metric(recon, target, mode=metricMode)
        standard_recon = scale_to_target(raw, target_raw, mode="mse")
        if(shiftFOV):
                standard_recon = shift_for_display(standard_recon)
        standard_mse, standard_psnr = compute_metric(standard_recon, target, mode="mse")
        return {
                "metric_mode": metricMode,
                "mse": mse,
                "psnr": psnr,
                "standard_mse": standard_mse,
                "standard_psnr": standard_psnr,
        }


# create instance of TI CGHGenerator and DeviceLibrary
G = CGHGenerator()
D = DeviceLibrary()

# Define input parameters
TargetImage = "./temporal_subset_inputs/Bear1_"
OutputDir = "./monomulti_temporal_output"
OutputRoot = Path(TargetImage).stem + "_532nm_tmux"
RandomSeed = 123

# Temporal multiplexing parameters.
# More frames and more iterations improve the summed-time reconstruction.
TemporalFrames = 24
NumIter = 300
TemporalSeedJitter = 2 * np.pi
LearningRate = 0.5
PixelGrouping = 1
Algorithm = "ADAM"
LossMode = "mse"
MetricMode = "mse"
RunSingleFrameBaseline = True
BaselineNumIter = NumIter

# testImage.bmp is a 1-bit halftone. This recovers the grayscale density image
# that the eye/camera sees when dots are averaged spatially and temporally.
PrepareHalftoneAsGrayscaleTarget = True
HalftoneBlurSigma = 1.5

# Device/image parameters.
DeviceName = "0.67"
UseFull067Width = True
PreserveAspect = False
ShiftFOV = True
FlipUD = True
FlipLR = False
InvertTarget = False
ColorChannel = 0
ShowImages = False
BinaryOutput = False
DisplayResult = False

if(RandomSeed is not None):
        np.random.seed(int(RandomSeed))
        if(torch is not None):
                torch.manual_seed(int(RandomSeed))

Path(OutputDir).mkdir(parents=True, exist_ok=True)
OptimizationTargetImage = TargetImage
if(PrepareHalftoneAsGrayscaleTarget):
        OptimizationTargetImage = prepare_density_target(
                TargetImage,
                Path(OutputDir) / (Path(TargetImage).stem + "_density_target.bmp"),
                blurSigma=HalftoneBlurSigma,
        )

DeviceDictionary = D.defineDevice(DeviceName, lutWavelength=532)
if(UseFull067Width and DeviceDictionary["device"] == "0.67"):
        DeviceDictionary["w"] = 1358
        DeviceDictionary["h"] = 800

# Call create CGH. Temporal ADAM jointly optimizes all frames so
# their summed/averaged intensity reconstructs the target image.
G.createCGH(DeviceDictionary=DeviceDictionary,
            filename=OptimizationTargetImage, bitPlanes=1, colorChannel=ColorChannel,
            FlipUD=FlipUD, FlipLR=FlipLR, InvertTarget=InvertTarget,
            alg=Algorithm, propMethod="Fourier", numIter=NumIter,
            showImages=ShowImages, initialPhase="Random", ShiftFOV=ShiftFOV,
            pixelGrouping=PixelGrouping, preserveAspect=PreserveAspect,
            algParams={
                    "temporalFrames": TemporalFrames,
                    "temporalSeedJitter": TemporalSeedJitter,
                    "learningRate": LearningRate,
                    "lossMode": LossMode,
                    "trainQuantized": False,
            })

output_template = Path(OutputDir) / (OutputRoot + ".bmp")
G.writeCGHToFile(str(output_template), binary=BinaryOutput)

metrics = compute_temporal_metrics(G, shiftFOV=ShiftFOV, metricMode=MetricMode)
baseline_metrics = None
if(RunSingleFrameBaseline):
        G_baseline = CGHGenerator()
        if(RandomSeed is not None):
                np.random.seed(int(RandomSeed))
                if(torch is not None):
                        torch.manual_seed(int(RandomSeed))
        G_baseline.createCGH(DeviceDictionary=DeviceDictionary,
                             filename=OptimizationTargetImage, bitPlanes=1, colorChannel=ColorChannel,
                             FlipUD=FlipUD, FlipLR=FlipLR, InvertTarget=InvertTarget,
                             alg=Algorithm, propMethod="Fourier", numIter=BaselineNumIter,
                             showImages=False, initialPhase="Random", ShiftFOV=ShiftFOV,
                             pixelGrouping=PixelGrouping, preserveAspect=PreserveAspect,
                             algParams={
                                     "temporalFrames": 1,
                                     "learningRate": LearningRate,
                                     "lossMode": LossMode,
                             })
        baseline_metrics = compute_single_frame_metrics(G_baseline, shiftFOV=ShiftFOV, metricMode=MetricMode)
integrated_preview_file = Path(OutputDir) / (OutputRoot + "_integrated_preview.bmp")
target_preview_file = Path(OutputDir) / (OutputRoot + "_target_preview.bmp")
metrics_file = Path(OutputDir) / (OutputRoot + "_metrics.txt")

save_normalized(integrated_preview_file, metrics["integrated_reconstruction"])
save_normalized(target_preview_file, metrics["target_display"])

with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("532 nm temporal multiplexing metrics\n")
        f.write("Metric mode: " + str(metrics["metric_mode"]) + "\n")
        f.write("Source image: " + TargetImage + "\n")
        f.write("Optimization target image: " + str(OptimizationTargetImage) + "\n")
        f.write("Temporal frames: " + str(TemporalFrames) + "\n")
        f.write("Iterations: " + str(NumIter) + "\n")
        f.write("Integrated summed-time MSE: " + str(metrics["integrated_mse"]) + "\n")
        f.write("Integrated summed-time PSNR: " + str(metrics["integrated_psnr"]) + "\n")
        f.write("Integrated standard MSE: " + str(metrics["standard_integrated_mse"]) + "\n")
        f.write("Integrated standard PSNR: " + str(metrics["standard_integrated_psnr"]) + "\n")
        if(baseline_metrics is not None):
                f.write("Single-frame baseline MSE: " + str(baseline_metrics["mse"]) + "\n")
                f.write("Single-frame baseline PSNR: " + str(baseline_metrics["psnr"]) + "\n")
                f.write("Single-frame baseline standard MSE: " + str(baseline_metrics["standard_mse"]) + "\n")
                f.write("Single-frame baseline standard PSNR: " + str(baseline_metrics["standard_psnr"]) + "\n")
                f.write("Temporal minus baseline PSNR: " + str(metrics["integrated_psnr"] - baseline_metrics["psnr"]) + "\n")
        f.write("Per-frame MSE:\n")
        for frame_idx, mse in enumerate(metrics["frame_mse"]):
                f.write("  Frame " + str(frame_idx).zfill(3) + ": " + str(mse) + "\n")
        f.write("Per-frame PSNR:\n")
        for frame_idx, psnr in enumerate(metrics["frame_psnr"]):
                f.write("  Frame " + str(frame_idx).zfill(3) + ": " + str(psnr) + "\n")

frame_files = [
        Path(OutputDir) / (OutputRoot + "_frame" + str(frame_idx).zfill(3) + ".bmp")
        for frame_idx in range(TemporalFrames)
]

print("Temporal multiplexed CGH complete.")
print("Wavelength: 532 nm")
print("Metric mode: " + str(metrics["metric_mode"]))
print("Integrated summed-time MSE: " + str(metrics["integrated_mse"]))
print("Integrated summed-time PSNR: " + str(metrics["integrated_psnr"]))
print("Integrated standard PSNR: " + str(metrics["standard_integrated_psnr"]))
if(baseline_metrics is not None):
        print("Single-frame baseline PSNR: " + str(baseline_metrics["psnr"]))
        print("Single-frame baseline standard PSNR: " + str(baseline_metrics["standard_psnr"]))
        print("Temporal minus baseline PSNR: " + str(metrics["integrated_psnr"] - baseline_metrics["psnr"]))
print("Per-frame PSNR:")
for frame_idx, psnr in enumerate(metrics["frame_psnr"]):
        print("  Frame " + str(frame_idx).zfill(3) + ": " + str(psnr))
print("Frame BMPs:")
for frame_file in frame_files:
        print("  " + str(frame_file))
print("Integrated preview: " + str(integrated_preview_file))
print("Metrics file: " + str(metrics_file))

# Optional: Display the summed-time reconstruction.
if(DisplayResult):
        cv2.destroyAllWindows()
        cv2.imshow("Temporal Integrated Image", display_normalize(metrics["integrated_reconstruction"]))
        cv2.waitKey(0)
