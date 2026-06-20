#!/usr/bin/env python3
"""
Convert input images to training data format:
- Read regular images from input folder
- Quantize to 1-bit bitmap
- Resize to 1345 x 800
- Generate paired hologram ground truth
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import argparse
from tqdm import tqdm
import cv2


def quantize_to_1bit(image_array):
    """
    Quantize image to 1-bit (binary) bitmap using Floyd-Steinberg dithering.
    
    Args:
        image_array: Input image as numpy array (grayscale, 0-255)
    
    Returns:
        Binary image (0 or 1)
    """
    # Work with float in range [0, 1]
    if image_array.dtype == np.uint8:
        img = image_array.astype(np.float32) / 255.0
    else:
        img = image_array.astype(np.float32)
    
    # Make a copy for dithering
    img = img.copy()
    height, width = img.shape
    
    # Floyd-Steinberg dithering
    for y in range(height):
        for x in range(width):
            old_val = img[y, x]
            new_val = 1.0 if old_val >= 0.5 else 0.0
            error = old_val - new_val
            
            # Distribute error to neighboring pixels
            if x + 1 < width:
                img[y, x + 1] += error * 7.0 / 16.0
            if y + 1 < height:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * 3.0 / 16.0
                img[y + 1, x] += error * 5.0 / 16.0
                if x + 1 < width:
                    img[y + 1, x + 1] += error * 1.0 / 16.0
            
            img[y, x] = new_val
    
    binary = (img > 0.5).astype(np.uint8)
    return binary


def resize_and_quantize_image(image_path, target_width=1358, target_height=800):
    """
    Load image, resize to target dimensions, and quantize to 1-bit.
    
    Args:
        image_path: Path to input image
        target_width: Target width in pixels (default: 1345)
        target_height: Target height in pixels (default: 800)
    
    Returns:
        1-bit binary image as numpy array
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize with aspect ratio preservation (add padding if needed)
    img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image
    resized_img = Image.new('L', (target_width, target_height), color=255)
    
    # Center the image
    offset_x = (target_width - img.width) // 2
    offset_y = (target_height - img.height) // 2
    resized_img.paste(img, (offset_x, offset_y))
    
    # Convert to numpy array and quantize to 1-bit
    img_array = np.array(resized_img, dtype=np.uint8)
    binary_img = quantize_to_1bit(img_array)
    
    return binary_img


def pack_bits(binary_array):
    """
    Pack 1-bit array into bytes for efficient storage.
    
    Args:
        binary_array: 2D binary array (height, width)
    
    Returns:
        Packed byte array (height, ceil(width/8))
        Original shape (height, width) for unpacking later
    """
    height, width = binary_array.shape
    
    # Pad width to multiple of 8
    pad_width = (8 - (width % 8)) % 8
    padded_width = width + pad_width
    
    # Pad the array
    padded = np.pad(binary_array, ((0, 0), (0, pad_width)), mode='constant')
    
    # Reshape and pack
    packed = np.packbits(padded, axis=1, bitorder='big')
    
    return packed, (height, width)


def process_input_folder(
    input_dir,
    output_dir,
    target_width=1345,
    target_height=800,
    patterns=None,
    verbose=True,
):
    """
    Process all images in input directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed data
        target_width: Target image width
        target_height: Target image height
        patterns: File patterns to search for (default: common image formats)
        verbose: Print progress
    """
    
    if patterns is None:
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for pattern in patterns:
        image_files.extend(input_path.glob(f'**/{pattern}'))
        image_files.extend(input_path.glob(f'**/{pattern.upper()}'))
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        print(f"   Searched for patterns: {', '.join(patterns)}")
        return []
    
    print(f"\n{'='*70}")
    print(f"Image Processing and Quantization")
    print(f"{'='*70}")
    print(f"Input directory:    {input_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Target resolution:  {target_width} x {target_height}")
    print(f"Output format:      1-bit bitmap (packed bytes)")
    print(f"Images to process:  {len(image_files)}")
    print(f"{'='*70}\n")
    
    processed_files = []
    failed_files = []
    
    iterator = tqdm(image_files, desc="Processing") if verbose else image_files
    
    for i, image_file in enumerate(iterator):
        try:
            # Process image
            binary_img = resize_and_quantize_image(
                image_file,
                target_width=target_width,
                target_height=target_height
            )
            
            # Create output filename
            output_name = f"image_{i:06d}.bmp"
            output_file = output_path / output_name
            
            # Save as BMP image
            bmp_img = Image.fromarray((binary_img * 255).astype(np.uint8), mode='L')
            bmp_img.save(output_file, format='BMP')
            
            file_size = output_file.stat().st_size
            processed_files.append({
                'input': str(image_file),
                'output': str(output_file),
                'size': binary_img.shape,
                'file_size': file_size
            })
            
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'size': f"{binary_img.shape[0]}x{binary_img.shape[1]}",
                    'file_size': f"{file_size/1024:.1f}KB"
                })
        
        except Exception as e:
            failed_files.append((image_file.name, str(e)))
            if verbose:
                tqdm.write(f"❌ Failed: {image_file.name} - {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Processed:  {len(processed_files)}/{len(image_files)}")
    
    if processed_files:
        total_size = sum(f['file_size'] for f in processed_files)
        print(f"Total size: {total_size / (1024**2):.1f} MB")
        print(f"Avg/file:   {total_size / len(processed_files) / 1024:.1f} KB")
    
    if failed_files:
        print(f"\n❌ Failed ({len(failed_files)}):")
        for name, error in failed_files[:5]:  # Show first 5 errors
            print(f"   {name}: {error}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
    
    print(f"{'='*70}\n")
    
    return processed_files


def verify_processed_data(output_dir, sample_count=3):
    """
    Verify processed data by loading and displaying sample stats.
    
    Args:
        output_dir: Directory containing processed BMP files
        sample_count: Number of samples to verify
    """
    output_path = Path(output_dir)
    bmp_files = sorted(output_path.glob('*.bmp'))[:sample_count]
    
    if not bmp_files:
        print("❌ No processed files found to verify")
        return
    
    print(f"\nVerifying {len(bmp_files)} sample(s)...\n")
    
    for bmp_file in bmp_files:
        img = Image.open(bmp_file)
        img_array = np.array(img, dtype=np.uint8)
        
        # Convert back to binary (0 or 1)
        binary_img = (img_array > 128).astype(np.uint8)
        
        print(f"File: {bmp_file.name}")
        print(f"  Image size:     {img.size[0]} x {img.size[1]}")
        print(f"  Image mode:     {img.mode}")
        print(f"  Unique values:  {np.unique(binary_img)}")
        print(f"  File size:      {bmp_file.stat().st_size / 1024:.1f} KB")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert input images to 1-bit quantized training data"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing images"
    )
    parser.add_argument(
        "-o", "--output",
        default="./training_data_1bit",
        help="Output directory for processed data (default: ./training_data_1bit)"
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=1345,
        help="Target image width (default: 1345)"
    )
    parser.add_argument(
        "-y", "--height",
        type=int,
        default=800,
        help="Target image height (default: 800)"
    )
    parser.add_argument(
        "-v", "--verify",
        action="store_true",
        help="Verify output after processing"
    )
    
    args = parser.parse_args()
    
    # Process images
    results = process_input_folder(
        input_dir=args.input_dir,
        output_dir=args.output,
        target_width=args.width,
        target_height=args.height,
        verbose=True,
    )
    
    # Verify if requested
    if args.verify and results:
        verify_processed_data(args.output, sample_count=min(3, len(results)))
