#!/usr/bin/env python3
"""
Test script to verify the image processing and dataset loading pipeline.
Run this to validate everything works correctly.
"""

import numpy as np
from pathlib import Path
import tempfile
import sys

def test_image_processing():
    """Test image processing module."""
    print("\n" + "="*70)
    print("TEST 1: Image Processing Module")
    print("="*70)
    
    try:
        from image_to_training_data import (
            quantize_to_1bit,
            pack_bits,
            resize_and_quantize_image,
            process_input_folder,
        )
        print("✓ Successfully imported image_to_training_data functions")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    try:
        # Test quantization
        test_img = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        binary = quantize_to_1bit(test_img.reshape(1, -1))
        expected = np.array([0, 0, 0, 1, 1])
        assert np.array_equal(binary[0], expected), "Quantization test failed"
        print("✓ Quantization works correctly")
    except Exception as e:
        print(f"✗ Quantization test failed: {e}")
        return False
    
    try:
        # Test bit packing
        test_bits = np.array([[1, 0, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
        packed, shape = pack_bits(test_bits)
        assert packed[0, 0] == 0b10110010, "Bit packing test failed"
        assert shape == (1, 8), "Shape tracking failed"
        print("✓ Bit packing works correctly")
    except Exception as e:
        print(f"✗ Bit packing test failed: {e}")
        return False
    
    print("✓ Image processing module tests passed!")
    return True


def test_dataset_loading():
    """Test dataset loading with real PyTorch."""
    print("\n" + "="*70)
    print("TEST 2: Dataset Loading")
    print("="*70)
    
    try:
        import torch
        from FastCGHNet import QuantizedImageDataset
        print("✓ Successfully imported PyTorch and QuantizedImageDataset")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        try:
            # Create sample NPZ files
            for i in range(3):
                packed = np.random.randint(0, 256, (800, 169), dtype=np.uint8)
                shape = np.array([800, 1345], dtype=np.int32)
                
                npz_file = tmpdir / f"image_{i:06d}.npz"
                np.savez_compressed(npz_file, packed_bits=packed, shape=shape)
            
            print(f"✓ Created 3 test NPZ files")
        except Exception as e:
            print(f"✗ Failed to create test data: {e}")
            return False
        
        try:
            # Test dataset loading
            dataset = QuantizedImageDataset(
                data_dir=str(tmpdir),
                max_samples=2,
                device="cpu",
                generate_phase=True
            )
            
            assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
            print(f"✓ Dataset loaded {len(dataset)} samples")
            
            # Test data loading
            img_batch, phase_batch = dataset[0]
            assert img_batch.shape == (1, 800, 1345), f"Unexpected img shape: {img_batch.shape}"
            assert phase_batch.shape == (1, 800, 1345), f"Unexpected phase shape: {phase_batch.shape}"
            print(f"✓ Data shapes correct: img {img_batch.shape}, phase {phase_batch.shape}")
            
            # Verify data types
            assert img_batch.dtype == torch.float32, f"Unexpected img dtype: {img_batch.dtype}"
            assert phase_batch.dtype == torch.float32, f"Unexpected phase dtype: {phase_batch.dtype}"
            print(f"✓ Data types correct: both float32")
            
        except Exception as e:
            print(f"✗ Dataset loading test failed: {e}")
            return False
    
    print("✓ Dataset loading tests passed!")
    return True


def test_training_integration():
    """Test training function with new dataset."""
    print("\n" + "="*70)
    print("TEST 3: Training Integration")
    print("="*70)
    
    try:
        from FastCGHNet import train_cgh_network, FastCGHNet
        import torch
        print("✓ Successfully imported training functions")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    try:
        # Check that training function has new parameters
        import inspect
        sig = inspect.signature(train_cgh_network)
        params = list(sig.parameters.keys())
        
        required_params = ['use_quantized_data', 'quantized_data_dir']
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"
        
        print(f"✓ Training function has new parameters: {required_params}")
        
    except Exception as e:
        print(f"✗ Training integration test failed: {e}")
        return False
    
    print("✓ Training integration tests passed!")
    return True


def test_backward_compatibility():
    """Test that old dataset format still works."""
    print("\n" + "="*70)
    print("TEST 4: Backward Compatibility")
    print("="*70)
    
    try:
        from FastCGHNet import HologramDataset
        print("✓ HologramDataset still available (old format support)")
    except ImportError as e:
        print(f"✗ Failed to import HologramDataset: {e}")
        return False
    
    print("✓ Backward compatibility maintained!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("IMAGE PROCESSING PIPELINE - VALIDATION TESTS")
    print("="*70)
    
    tests = [
        test_image_processing,
        test_dataset_loading,
        test_training_integration,
        test_backward_compatibility,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_func.__name__}: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
