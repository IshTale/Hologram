#!/usr/bin/env python3
"""
Batch process images with FastCGHNet
Convert entire folders instantly
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, '/Users/Ish/Hologram')
from fast_predict import fast_predict


def batch_process(
    input_dir,
    output_dir,
    model_path="/Users/Ish/Hologram/models/best_model.pt",
    pattern="*.bmp",
    verbose=True,
):
    """
    Process all images in a directory
    
    Args:
        input_dir: Input image directory
        output_dir: Output CGH directory
        model_path: Trained model path
        pattern: Glob pattern for files
        verbose: Show progress
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = sorted(input_path.glob(pattern))
    
    if not files:
        print(f"❌ No files matching '{pattern}' in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"FastCGHNet Batch Processing")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model:            {model_path}")
    print(f"Files to process: {len(files)}")
    print(f"{'='*60}\n")
    
    # Process each file
    times = []
    failed = []
    
    iterator = tqdm(files, desc="Processing") if verbose else files
    
    for img_file in iterator:
        try:
            # Output filename
            output_file = output_path / (img_file.stem + "_cgh.bmp")
            
            # Predict
            t0 = time.time()
            fast_predict(
                str(img_file),
                model_path=model_path,
                output_bmp=str(output_file),
                output_phase=None,
            )
            t_elapsed = time.time() - t0
            times.append(t_elapsed)
            
            if verbose:
                iterator.set_postfix({
                    'time': f'{t_elapsed*1000:.0f}ms',
                    'avg': f'{sum(times)/len(times)*1000:.0f}ms'
                })
        
        except Exception as e:
            failed.append((img_file.name, str(e)))
            if verbose:
                tqdm.write(f"❌ Failed: {img_file.name} - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed:  {len(files) - len(failed)}/{len(files)}")
    
    if times:
        print(f"Total time: {sum(times):.1f}s")
        print(f"Avg/file:   {sum(times)/len(times)*1000:.0f}ms")
        print(f"Throughput: {len(times)/sum(times):.1f} images/sec")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name, err in failed:
            print(f"   {name}: {err}")
    
    print(f"{'='*60}\n")
    
    return len(files) - len(failed), len(files)


def process_training_samples(model_path=None, max_samples=None):
    """Quick utility to process all training data samples"""
    
    if model_path is None:
        model_path = "/Users/Ish/Hologram/models/best_model.pt"
    
    training_dir = Path("/Users/Ish/Hologram/Training Data/samples")
    output_dir = Path("/Users/Ish/Hologram/output/training_holograms")
    
    # Get sample directories
    samples = sorted([d for d in training_dir.iterdir() if d.is_dir()])
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"\nProcessing {len(samples)} training samples...")
    print(f"Output: {output_dir}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    times = []
    for sample_dir in tqdm(samples, desc="Training samples"):
        img_file = sample_dir / "view_1.bmp"
        if not img_file.exists():
            continue
        
        output_file = output_dir / (sample_dir.name + "_cgh.bmp")
        
        t0 = time.time()
        try:
            fast_predict(
                str(img_file),
                model_path=model_path,
                output_bmp=str(output_file),
            )
            times.append(time.time() - t0)
        except Exception as e:
            tqdm.write(f"Failed {sample_dir.name}: {e}")
    
    print(f"\n✓ Processed {len(times)} samples")
    if times:
        print(f"  Total: {sum(times):.1f}s")
        print(f"  Avg:   {sum(times)/len(times)*1000:.0f}ms/image")
        print(f"  Rate:  {len(times)/sum(times):.1f} images/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images with FastCGHNet")
    parser.add_argument("--input", help="Input directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--model", default="/Users/Ish/Hologram/models/best_model.pt", help="Model path")
    parser.add_argument("--pattern", default="*.bmp", help="File glob pattern")
    parser.add_argument("--training-samples", action="store_true", help="Process training data")
    parser.add_argument("--max", type=int, help="Max samples to process")
    
    args = parser.parse_args()
    
    if args.training_samples:
        process_training_samples(model_path=args.model, max_samples=args.max)
    else:
        if not args.input or not args.output:
            parser.print_help()
            sys.exit(1)
        batch_process(args.input, args.output, args.model, args.pattern)
