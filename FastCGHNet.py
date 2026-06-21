"""
FastCGHNet: Neural network for direct hologram phase prediction
Replaces iterative optimization with a single forward pass (~10-50ms per image)

Supports:
- GPU training (CUDA)
- CPU training
- AMD KRIA deployment (ARM NEON, quantized inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import cv2
import platform
import os
import time

# Platform detection
IS_ARM = platform.machine() in ['armv7l', 'aarch64', 'arm64']
IS_KRIA = IS_ARM and os.path.exists('/proc/device-tree/model')


class FastCGHNet(nn.Module):
    """
    Simple dense encoder for direct image-to-phase hologram prediction.
    No pooling - maintains full resolution throughout.
    Input: grayscale image (1, 800, 1358)
    Output: phase hologram (1, 800, 1358) in range [0, 2π]
    """
    def __init__(self, in_channels=1, out_channels=1, lite=False):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            lite: If True, use reduced model for ARM/KRIA deployment
        """
        super().__init__()
        c = 8 if lite else 16
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c, c*2, 3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c*2, c*4 if not lite else c*2, 3, padding=1),
            nn.BatchNorm2d(c*4 if not lite else c*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c*4 if not lite else c*2, c*4 if not lite else c*2, 3, padding=1),
            nn.BatchNorm2d(c*4 if not lite else c*2),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(c*4 if not lite else c*2, c*2, 3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c*2, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(c, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x * (2 * np.pi)


class HologramDataset(Dataset):
    """Load training samples from disk (old format with phase ground truth)"""
    def __init__(self, samples_dir, max_samples=None, device='cpu'):
        self.samples_dir = Path(samples_dir)
        self.device = device
        self.sample_dirs = sorted([d for d in self.samples_dir.iterdir() if d.is_dir()])
        if max_samples:
            self.sample_dirs = self.sample_dirs[:max_samples]
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # Load input (binary image packed as bits)
        input_file = sample_dir / "input_views_packed.npz"
        data = np.load(input_file)
        packed_bits = data['packed_bits']
        shape = tuple(data['shape'])  # (num_views, height, width)
        
        # Unpack bits to get binary image
        if packed_bits.ndim == 3 and shape[0] == 2:  # Two views - take first one
            bits_view1 = packed_bits[0]
        else:
            bits_view1 = packed_bits
        
        # Unpack bytes to bits - bits_view1 is (height, width_bytes)
        height, width_bytes = bits_view1.shape
        img_binary_unpacked = np.unpackbits(bits_view1, axis=1, bitorder='big')
        img_binary = img_binary_unpacked.reshape(height, width_bytes * 8)[:, :shape[2]]
        img_binary = img_binary.astype(np.float32)
        
        # Load ground truth phase
        phase_file = sample_dir / "cgh_phase_cont_float32.npy"
        phase_gt = np.load(phase_file).astype(np.float32)
        
        # Add channel dimension
        img_input = torch.from_numpy(img_binary[np.newaxis, :, :]).to(self.device)
        phase_target = torch.from_numpy(phase_gt[np.newaxis, :, :]).to(self.device)
        
        return img_input, phase_target


class QuantizedImageDataset(Dataset):
    """
    Load 1-bit quantized images from NPZ files.
    Images are resized to 1345x800 and quantized to binary.
    Phase ground truth is generated using simulated holographic diffraction.
    """
    def __init__(self, data_dir, max_samples=None, device='cpu', generate_phase=True):
        """
        Args:
            data_dir: Directory containing .npz files with quantized images
            max_samples: Maximum number of samples to load
            device: torch device
            generate_phase: Whether to generate synthetic phase data (default: True)
        """
        self.data_dir = Path(data_dir)
        self.device = device
        self.generate_phase = generate_phase
        
        # Find all NPZ files
        self.npz_files = sorted(self.data_dir.glob('*.npz'))
        if max_samples:
            self.npz_files = self.npz_files[:max_samples]
        
        if not self.npz_files:
            raise ValueError(f"No .npz files found in {data_dir}")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        
        # Load quantized image
        data = np.load(npz_file)
        packed_bits = data['packed_bits']  # (height, width_bytes)
        shape = tuple(data['shape'])  # (height, width)
        
        # Unpack bytes to bits
        height, width_bytes = packed_bits.shape
        img_binary_unpacked = np.unpackbits(packed_bits, axis=1, bitorder='big')
        img_binary = img_binary_unpacked.reshape(height, width_bytes * 8)[:, :shape[1]]
        img_binary = img_binary.astype(np.float32)
        
        # Generate synthetic phase ground truth
        if self.generate_phase:
            # Simple approach: random phase with some correlation to image structure
            phase_gt = self._generate_synthetic_phase(img_binary, shape)
        else:
            # Use random phase if no generation requested
            phase_gt = np.random.rand(*shape).astype(np.float32) * (2 * np.pi)
        
        # Add channel dimension
        img_input = torch.from_numpy(img_binary[np.newaxis, :, :]).to(self.device)
        phase_target = torch.from_numpy(phase_gt[np.newaxis, :, :]).to(self.device)
        
        return img_input, phase_target
    
    def _generate_synthetic_phase(self, img_binary, shape):
        """
        Generate synthetic phase based on image structure.
        
        This is a placeholder that generates phase with some correlation to the image.
        In production, this could use actual hologram simulation.
        """
        height, width = shape
        
        # Base random phase
        phase = np.random.rand(height, width).astype(np.float32) * (2 * np.pi)
        
        # Add some structure based on image
        img_smooth = cv2.GaussianBlur(img_binary, (21, 21), 0)
        phase += img_smooth * np.pi  # Modulate phase by image content
        
        # Normalize to [0, 2π]
        phase = np.mod(phase, 2 * np.pi)
        
        return phase


def train_cgh_network(
    samples_dir="/Users/Ish/Hologram/Training Data/samples",
    output_dir="/Users/Ish/Hologram/models",
    num_epochs=30,
    batch_size=2,
    learning_rate=5e-4,
    max_samples=None,
    use_quantized_data=False,
    quantized_data_dir=None,
    lite=False,
):
    """
    Train FastCGHNet on hologram data.
    
    Args:
        samples_dir: Directory with pre-processed samples (old format)
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_samples: Maximum samples to load (None = all)
        use_quantized_data: If True, use QuantizedImageDataset instead
        quantized_data_dir: Directory with 1-bit quantized images (required if use_quantized_data=True)
        lite: If True, train lightweight model for ARM deployment
    """
    
    Path(output_dir).mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FastCGHNet(in_channels=1, out_channels=1, lite=lite).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if lite:
        print("(Using lite model for KRIA/ARM deployment)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print("Loading dataset...")
    if use_quantized_data:
        if quantized_data_dir is None:
            raise ValueError("quantized_data_dir must be specified when use_quantized_data=True")
        dataset = QuantizedImageDataset(
            data_dir=quantized_data_dir,
            max_samples=max_samples,
            device=device,
            generate_phase=True
        )
        print(f"Using QuantizedImageDataset from: {quantized_data_dir}")
    else:
        dataset = HologramDataset(samples_dir, max_samples=max_samples, device=device)
        print(f"Using HologramDataset from: {samples_dir}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for img_batch, phase_batch in pbar:
            optimizer.zero_grad()
            
            phase_pred = model(img_batch)
            
            loss = criterion(phase_pred, phase_batch)
            epoch_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.8f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'lite': lite,
            }
            model_name = "best_model_lite.pt" if lite else "best_model.pt"
            checkpoint_path = Path(output_dir) / model_name
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Saved best model to {checkpoint_path}")
    
    print("Training complete!")
    return model


def export_onnx(
    model_path="/Users/Ish/Hologram/models/best_model.pt",
    output_path="/Users/Ish/Hologram/models/fastcghnet.onnx",
    lite=False,
):
    """
    Export PyTorch model to ONNX for KRIA/ARM deployment
    
    ONNX can be deployed with:
    - ONNX Runtime (lightweight, cross-platform)
    - Vitis AI (AMD FPGA acceleration)
    - TVM (multi-backend compiler)
    """
    import sys
    
    try:
        import onnx
    except ImportError:
        print("Installing onnx...")
        os.system(f"{sys.executable} -m pip install onnx -q")
        import onnx
    
    device = torch.device("cpu")
    
    model = FastCGHNet(in_channels=1, out_channels=1, lite=lite).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Exporting model from {model_path} to ONNX...")
    
    dummy_input = torch.randn(1, 1, 800, 1358, device=device)
    
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['image'],
            output_names=['phase'],
            opset_version=13,
            dynamic_axes={
                'image': {0: 'batch_size'},
                'phase': {0: 'batch_size'}
            },
            verbose=False,
        )
    
    print(f"✓ Exported to {output_path}")
    print(f"  Model can be deployed on KRIA using:")
    print(f"    - ONNX Runtime: pip install onnxruntime")
    print(f"    - Vitis AI: AMD FPGA quantization + deployment")
    print(f"    - TVM: Apache TVM compiler for optimization")
    
    return output_path


def predict_hologram_onnx(
    image_path,
    onnx_model_path="/Users/Ish/Hologram/models/fastcghnet.onnx",
    output_path="/Users/Ish/Hologram/output/cgh_fast.bmp",
):
    """
    Predict using ONNX Runtime (cross-platform, efficient on ARM)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Installing onnxruntime...")
        os.system("pip install onnxruntime -q")
        import onnxruntime as ort
    
    from PIL import Image
    from PLM import DeviceLibrary, CGHGenerator
    
    session = ort.InferenceSession(onnx_model_path)
    
    img_pil = Image.open(image_path).convert('L')
    img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    
    if img_array.shape != (800, 1358):
        img_array = cv2.resize(img_array, (1358, 800))
    
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    
    img_input = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
    
    t0 = time.time()
    outputs = session.run(None, {'image': img_input})
    t_pred = time.time() - t0
    
    phase_np = outputs[0].squeeze()
    
    print(f"✓ ONNX Prediction time: {t_pred*1000:.1f}ms")
    print(f"  Phase range: [{phase_np.min():.3f}, {phase_np.max():.3f}]")
    
    device_lib = DeviceLibrary()
    device_dict = device_lib.defineDevice("0.67")
    
    phase_disc, state_disc = CGHGenerator.discretePhase(
        phase_np, device_dict["nLevel"], device_dict["pLevel"]
    )
    cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
    
    cgh_uint8 = cv2.normalize(cgh_mapped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path, cgh_uint8)
    
    print(f"✓ Wrote CGH to: {output_path}")
    
    return phase_np, cgh_mapped


def predict_hologram(
    image_path,
    model_path="/Users/Ish/Hologram/models/best_model.pt",
    output_path="/Users/Ish/Hologram/output/cgh_fast.bmp",
    use_onnx=None,
    lite=False,
):
    """
    Fast hologram prediction using trained model
    Input: image file path
    Output: hologram phase and mapped CGH
    
    Args:
        image_path: Input image
        model_path: PyTorch model path
        output_path: Output BMP path
        use_onnx: If True, use ONNX Runtime (auto-detect if None and on ARM)
        lite: Use lightweight model
    """
    from PIL import Image
    from PLM import DeviceLibrary, CGHGenerator
    
    if use_onnx is None:
        use_onnx = IS_ARM
    
    if use_onnx:
        onnx_path = model_path.replace('.pt', '.onnx')
        if Path(onnx_path).exists():
            return predict_hologram_onnx(image_path, onnx_path, output_path)
        else:
            print(f"⚠️  ONNX model not found at {onnx_path}")
            print(f"Export with: export_onnx('{model_path}')")
            use_onnx = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FastCGHNet(lite=lite).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    img_pil = Image.open(image_path).convert('L')
    img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    
    if img_array.shape != (800, 1358):
        img_array = cv2.resize(img_array, (1358, 800))
    
    img_tensor = torch.from_numpy(img_array[np.newaxis, np.newaxis, :, :]).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        phase_pred = model(img_tensor)
    t_pred = time.time() - t0
    
    phase_np = phase_pred.squeeze().cpu().numpy()
    
    print(f"Prediction time: {t_pred*1000:.1f}ms")
    print(f"Phase range: [{phase_np.min():.3f}, {phase_np.max():.3f}]")
    
    device_lib = DeviceLibrary()
    device_dict = device_lib.defineDevice("0.67")
    
    phase_disc, state_disc = CGHGenerator.discretePhase(
        phase_np, device_dict["nLevel"], device_dict["pLevel"]
    )
    
    cgh_mapped = device_lib.formatPLM(device_dict, state_disc)
    
    cgh_uint8 = cv2.normalize(cgh_mapped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path, cgh_uint8)
    
    print(f"Wrote CGH to: {output_path}")
    
    return phase_np, cgh_mapped


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_cgh_network(
            num_epochs=30,
            batch_size=2,
            max_samples=None,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/Ish/Hologram/models/best_model.pt"
        export_onnx(model_path)
    else:
        print("FastCGHNet model ready.")
        print("Usage:")
        print("  python FastCGHNet.py train              - Train model")
        print("  python FastCGHNet.py export <model.pt> - Export to ONNX")
