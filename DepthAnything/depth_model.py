import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoProcessor


def _load_model_assets(repo_id):
    try:
        processor = AutoImageProcessor.from_pretrained(repo_id, local_files_only=False)
        model = AutoModelForDepthEstimation.from_pretrained(repo_id, local_files_only=False)
        return processor, model
    except Exception as e:
        print(f"Warning: failed to load HF model assets from {repo_id}: {e}")
        try:
            processor = AutoProcessor.from_pretrained(repo_id, local_files_only=False)
            model = AutoModelForDepthEstimation.from_pretrained(repo_id, local_files_only=False)
            return processor, model
        except Exception as inner_e:
            raise RuntimeError(
                f"Unable to load Depth Anything V2 from '{repo_id}'. "
                "Ensure you have network access, a valid HF_TOKEN if needed, "
                "and no local directory named 'depth-anything' or 'Depth-Anything-V2-Small-hf' "
                "in your current working directory. Original error: {inner_e}"
            ) from inner_e


class DepthEstimator:
    def __init__(self, use_amp=False):
        print("Loading Depth Anything V2...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = use_amp and self.device == "cuda"

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.processor, self.model = _load_model_assets(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Running on: {self.device}")

    def estimate_depth(self, frame):
        original_h, original_w = frame.shape[:2]

        small_frame = cv2.resize(frame, (384, 384))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        depth = outputs.predicted_depth.squeeze().detach().cpu().float().numpy()

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        depth_min = float(depth.min())
        depth_max = float(depth.max())

        if depth_max - depth_min < 1e-6:
            depth_uint8 = np.zeros_like(depth, dtype=np.uint8)
        else:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

        depth_uint8 = cv2.resize(depth_uint8, (original_w, original_h))

        return depth_uint8