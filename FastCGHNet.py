import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import cv2
import os
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SpectralConv2d(nn.Module):
    """
    Fourier Neural Operator spectral convolution layer.
    Performs convolution in the frequency domain using learnable complex weights.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def complex_mult2d(self, input, weight):
        return torch.einsum("bixy,ioxy->boxy", input, weight)

    def forward(self, x):
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.complex_mult2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.complex_mult2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Pad to full output size and inverse transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        # If output channels differ from input, handle dimension mismatch
        if x.shape[1] != self.out_channels:
            x = x[:, :self.out_channels, :, :]
        
        return x


class FNOBlock(nn.Module):
    """
    FNO block combining spectral and spatial convolutions with residual connection.
    Spectral path captures long-range frequency dependencies; spatial path acts as skip connection.
    """
    def __init__(self, channels, modes1=16, modes2=16):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x_spectral = self.spectral_conv(x)
        x_spatial = self.spatial_conv(x)
        return self.activation(self.norm(x_spectral + x_spatial))


class FastCGHNet(nn.Module):
    """
    U-Net for direct image-to-phase hologram prediction.
    Output is a continuous, unbounded phase map (radians). We do NOT bound it with
    a Sigmoid because exp(1j*phase) is periodic, so any real value is a valid phase.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck with FNO (Fourier Neural Operator)
        # At this point in the encoder, feature map is (25, 42), so use modes1=12, modes2=20
        bottleneck_channels = features[-1] * 2
        self.channel_expand = nn.Conv2d(features[-1], bottleneck_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            FNOBlock(bottleneck_channels, modes1=12, modes2=20),
            FNOBlock(bottleneck_channels, modes1=12, modes2=20),
            FNOBlock(bottleneck_channels, modes1=12, modes2=20)
        )

        # Up part of UNET
        for i in range(len(features)):
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.ups.append(nn.Conv2d(features[-1-i]*2, features[-1-i], kernel_size=1))  # Channel reduction
            self.ups.append(DoubleConv(features[-1-i]*2, features[-1-i]))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.channel_expand(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling and skip connections
        for i in range(0, len(self.ups), 3):  # 3 steps: Upsample, Conv(1x1), DoubleConv
            upsample = self.ups[i]
            channel_reduce = self.ups[i+1]
            double_conv = self.ups[i+2]

            x = upsample(x)
            x = channel_reduce(x)

            skip_connection = skip_connections[i//3]

            # Handle odd resolutions (1358 is not divisible by 16)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = double_conv(concat_skip)

        return self.final_conv(x)


class HologramDataset(Dataset):
    """
    Loads the binary processed image from each sample.

    For the physics (reconstruction) loss we DO NOT need the stored
    cgh_phase_cont_float32.npy at all: the target the network must reproduce is the
    image itself. So __getitem__ returns (input_image, target_image) where both are
    the same 1-channel binary image in [0, 1]. This makes training self-supervised
    and removes the dependence on the (non-unique, random) ADAM phase labels.
    """
    def __init__(self, samples_dir, max_samples=None, device='cpu', use_cache=True):
        self.samples_dir = Path(samples_dir)
        self.device = device
        self.use_cache = use_cache
        self.cache = {}
        self.sample_dirs = sorted([d for d in self.samples_dir.iterdir() if d.is_dir()])
        if max_samples:
            self.sample_dirs = self.sample_dirs[:max_samples]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        sample_dir = self.sample_dirs[idx]

        # Load input (binary image packed as bits): shape (1, H, W_bytes)
        data = np.load(sample_dir / "input_views_packed.npz")
        packed_bits = data['packed_bits']
        original_hw_shape = tuple(data['shape'])            # (H, W)

        single_view_packed_bits = packed_bits.squeeze(axis=0)   # (H, W_bytes)
        height, width_bytes = single_view_packed_bits.shape
        original_width = original_hw_shape[1]

        img_binary_unpacked = np.unpackbits(single_view_packed_bits, axis=1, bitorder='big')
        img_binary = img_binary_unpacked.reshape(height, width_bytes * 8)[:, :original_width]
        img_binary = img_binary.astype(np.float32)          # values in {0.0, 1.0}

        img_input = torch.from_numpy(img_binary[np.newaxis, :, :])   # (1, H, W)
        target_img = img_input.clone()                               # reconstruct the same image

        sample = (img_input, target_img)
        if self.use_cache:
            self.cache[idx] = sample
        return sample


def reconstruction_loss(pred_phase, target_img, shift_fov=True, eps=1e-20):
    """
    Physics-based CGH loss.

    Propagate the predicted phase to the image plane with the SAME Fourier forward
    model used by the optimizer (PLM.torchProp('forward') == ifft2), then match the
    reconstructed intensity to the desired image. Any valid hologram is rewarded, so
    unlike phase regression this objective is well-posed and the loss actually descends.

    pred_phase : (B, 1, H, W) real radians (unbounded; exp(1j*phase) is periodic)
    target_img : (B, 1, H, W) desired intensity in [0, 1]
    """
    field = torch.exp(1j * pred_phase.float())            # unit-amplitude hologram
    recon = torch.fft.ifft2(field)                        # forward propagation
    intensity = recon.real ** 2 + recon.imag ** 2         # reconstructed |.|^2

    t = target_img.float()
    if shift_fov:                                         # match resizeTarget(ShiftFOV=True)
        t = torch.roll(t, shifts=(t.shape[-2] // 2, t.shape[-1] // 2), dims=(-2, -1))

    # Per-image least-squares brightness gain == optimizer's _torchWeightedGain.
    num = (intensity * t).mean(dim=(-1, -2), keepdim=True)
    den = (intensity * intensity).mean(dim=(-1, -2), keepdim=True) + eps
    s = num / den
    return torch.mean((s * intensity - t) ** 2)


def train_cgh_network(
    samples_dir="/content/drive/MyDrive/New Training Data/samples",
    output_dir="/content/drive/MyDrive/New Training Data/models",
    num_epochs=30,
    batch_size=8,
    learning_rate=5e-4,
    max_samples=None,
    shift_fov=True,
):
    """
    Train FastCGHNet with the physics (reconstruction) loss.

    The network predicts a continuous phase; the loss reconstructs the image from that
    phase and compares it to the target image. The stored ADAM phase labels are NOT used.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_in_channels = 1  # single-channel binary image in

    model = FastCGHNet(in_channels=model_in_channels, out_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("Loading dataset...")
    dataset = HologramDataset(samples_dir, max_samples=max_samples, device='cpu')
    print(f"Using HologramDataset from: {samples_dir}")

    num_workers = min(4, os.cpu_count() or 2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Using {num_workers} DataLoader workers.")

    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for img_batch, target_batch in pbar:
            img_batch = img_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            phase_pred = model(img_batch)                       # (B, 1, H, W) predicted phase
            loss = reconstruction_loss(phase_pred, target_batch, shift_fov=shift_fov)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            psnr = -10.0 * math.log10(loss.item() + 1e-12)     # reconstruction PSNR (higher = better)
            epoch_psnr += psnr
            pbar.set_postfix({'loss': loss.item(), 'psnr_dB': psnr})

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB, "
              f"LR: {scheduler.get_last_lr()[0]:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'model_in_channels': model_in_channels,
                'shift_fov': shift_fov,
            }
            checkpoint_path = Path(output_dir) / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Saved best model to {checkpoint_path}")

    print("Training complete!")
    return model


def predict_hologram(
    image_path,
    model_path="/content/drive/MyDrive/New Training Data/models/best_model.pt",
    output_path="/content/drive/MyDrive/New Training Data/output/cgh_fast.bmp",
    model_in_channels=1,
):
    """
    Fast hologram prediction: image -> predicted continuous phase -> 16-level
    discretisation -> PLM electrode bitmap (BMP). The BMP is a deterministic
    post-processing of the predicted phase; the network never regresses the BMP.
    """
    from PIL import Image
    from PLM import DeviceLibrary, CGHGenerator
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastCGHNet(in_channels=model_in_channels).to(device)
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

    import time
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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cgh_uint8 = cv2.normalize(cgh_mapped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path, cgh_uint8)
    print(f"Wrote CGH to: {output_path}")

    return phase_np, cgh_mapped
