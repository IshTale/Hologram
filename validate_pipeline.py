#!/usr/bin/env python3
"""
Validation script for the image processing pipeline.
Checks that all modules are properly structured and integrated.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ Missing: {description}")
        return False


def check_function_exists(module_name, function_name, description):
    """Check if a function exists in a module."""
    try:
        module = __import__(module_name)
        if hasattr(module, function_name):
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ Missing function: {description}")
            return False
    except ImportError:
        print(f"✗ Cannot import {module_name}: {description}")
        return False


def check_class_exists(module_name, class_name, description):
    """Check if a class exists in a module."""
    try:
        if '.' in module_name:
            parts = module_name.rsplit('.', 1)
            module = __import__(parts[0], fromlist=[parts[1]])
        else:
            module = __import__(module_name)
        
        if hasattr(module, class_name):
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ Missing class: {description}")
            return False
    except Exception as e:
        print(f"✗ Cannot import {module_name}: {e}")
        return False


def main():
    """Run validation checks."""
    print("\n" + "="*70)
    print("IMAGE PROCESSING PIPELINE - STRUCTURE VALIDATION")
    print("="*70)
    
    results = []
    
    # Check files
    print("\n1. Checking Files")
    print("-" * 70)
    files_to_check = [
        ("image_to_training_data.py", "Image processing script"),
        ("FastCGHNet.py", "Neural network and datasets"),
        ("QUICKSTART_GUIDE.md", "Quick-start documentation"),
        ("IMAGE_PROCESSING_README.md", "Technical documentation"),
        ("test_pipeline.py", "Validation tests"),
    ]
    
    for filepath, description in files_to_check:
        results.append(check_file_exists(filepath, description))
    
    # Check functions in image_to_training_data.py
    print("\n2. Checking Image Processing Functions")
    print("-" * 70)
    
    functions_to_check = [
        ("image_to_training_data", "quantize_to_1bit", "1-bit quantization function"),
        ("image_to_training_data", "resize_and_quantize_image", "Resize and quantize function"),
        ("image_to_training_data", "pack_bits", "Bit packing function"),
        ("image_to_training_data", "process_input_folder", "Batch processing function"),
        ("image_to_training_data", "verify_processed_data", "Data verification function"),
    ]
    
    for module, func, desc in functions_to_check:
        results.append(check_function_exists(module, func, desc))
    
    # Check classes in FastCGHNet.py
    print("\n3. Checking FastCGHNet Classes")
    print("-" * 70)
    
    classes_to_check = [
        ("FastCGHNet", "FastCGHNet", "Neural network model"),
        ("FastCGHNet", "HologramDataset", "Original dataset loader (backward compatible)"),
        ("FastCGHNet", "QuantizedImageDataset", "New 1-bit quantized image dataset"),
    ]
    
    for module, cls, desc in classes_to_check:
        results.append(check_class_exists(module, cls, desc))
    
    # Check function signatures
    print("\n4. Checking Training Function Parameters")
    print("-" * 70)
    
    try:
        import inspect
        from FastCGHNet import train_cgh_network
        
        sig = inspect.signature(train_cgh_network)
        params = list(sig.parameters.keys())
        
        required_params = ['use_quantized_data', 'quantized_data_dir']
        all_present = all(p in params for p in required_params)
        
        if all_present:
            print(f"✓ New parameters in train_cgh_network: {required_params}")
            results.append(True)
        else:
            missing = [p for p in required_params if p not in params]
            print(f"✗ Missing parameters: {missing}")
            results.append(False)
    except Exception as e:
        print(f"✗ Cannot check function signature: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All components validated successfully!")
        print("\nYou can now:")
        print("  1. Process images: python image_to_training_data.py <input_dir> -o <output_dir>")
        print("  2. Train network: python -c \"from FastCGHNet import train_cgh_network\"")
        print("  3. Read the guide: IMAGE_PROCESSING_README.md or QUICKSTART_GUIDE.md")
        return 0
    else:
        print(f"\n✗ {total - passed} validation(s) failed")
        print("\nPlease check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
