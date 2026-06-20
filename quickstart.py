#!/usr/bin/env python3
"""
Quick-start example: Process images and train FastCGHNet
Shows the complete workflow from raw images to trained model.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"✓ Complete: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete workflow: Process images and train FastCGHNet"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output",
        default="./training_data_1bit",
        help="Output directory for processed data (default: ./training_data_1bit)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum images to process (default: all)"
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip image processing (if already done)"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (only process images)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    steps_completed = 0
    total_steps = 2 if not args.skip_train else 1
    
    # Step 1: Process images
    if not args.skip_process:
        cmd = f"python image_to_training_data.py '{args.input_dir}' -o '{args.output}' -v"
        if args.max_samples:
            # Note: max_samples not supported in the script, but shown for reference
            pass
        
        if not run_command(cmd, "Process images to 1-bit quantized format"):
            sys.exit(1)
        steps_completed += 1
    else:
        print(f"\n✓ Skipped: Image processing (using existing data)")
        steps_completed += 1
    
    # Step 2: Train network
    if not args.skip_train:
        # Create Python training script inline
        train_script = """
import sys
sys.path.insert(0, '/Users/Ish/Hologram')

from FastCGHNet import train_cgh_network

print("\\n" + "="*70)
print("TRAINING FASTCGHNET")
print("="*70)

train_cgh_network(
    use_quantized_data=True,
    quantized_data_dir='{output_dir}',
    num_epochs={epochs},
    batch_size={batch_size},
    output_dir='/Users/Ish/Hologram/models'
)
""".format(
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Write and run training script
        script_file = Path("_train_temp.py")
        script_file.write_text(train_script)
        
        cmd = f"python {script_file}"
        if not run_command(cmd, "Train FastCGHNet on quantized images"):
            script_file.unlink()
            sys.exit(1)
        
        script_file.unlink()
        steps_completed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"WORKFLOW COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed data saved to: {args.output}")
    print(f"Trained model saved to: /Users/Ish/Hologram/models/best_model.pt")
    print(f"{'='*70}\n")
    
    # Next steps
    print("Next steps:")
    if not args.skip_process:
        print(f"  1. ✓ Processed {args.input_dir} to 1-bit quantized format")
    if not args.skip_train:
        print(f"  2. ✓ Trained model for {args.epochs} epochs")
    print(f"\nYou can now:")
    print(f"  • Evaluate the model: python fast_predict.py <image> -m models/best_model.pt")
    print(f"  • Batch process images: python batch_process.py <input_dir> <output_dir>")
    print(f"  • Fine-tune: python {parser.prog} <more_images> --epochs 5")
    print()


if __name__ == "__main__":
    main()
