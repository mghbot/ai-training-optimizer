#!/usr/bin/env python3
"""
Validation script to check ACMDS installation and dependencies
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✓ CUDA available: {device_count} device(s)")
            print(f"  - Device 0: {device_name}")
            print(f"  - CUDA version: {cuda_version}")
            return True
        else:
            print("⚠ CUDA not available (CPU only mode)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def validate_acmds():
    """Validate ACMDS components"""
    try:
        from acmds_optimizer import (
            CurriculumConfig,
            MemoryPressureController,
            DifficultyAwareDataset,
            AdaptiveCurriculumSampler,
            RTX3090ACMDSTrainer
        )
        print("✓ ACMDS optimizer components")

        # Test config creation
        config = CurriculumConfig()
        print("✓ CurriculumConfig instantiation")

        return True
    except Exception as e:
        print(f"✗ ACMDS validation failed: {e}")
        return False

def main():
    """Run all validation checks"""
    print("=" * 60)
    print("ACMDS Installation Validation")
    print("=" * 60)

    print("\nChecking core dependencies:")
    all_good = True

    # Core dependencies
    all_good &= check_import("torch", "PyTorch")
    all_good &= check_import("numpy", "NumPy")
    all_good &= check_import("psutil", "psutil")

    # Optional dependencies
    print("\nChecking optional dependencies:")
    check_import("torchvision", "torchvision")
    check_import("pytest", "pytest")

    # CUDA check
    print("\nChecking CUDA:")
    check_cuda()

    # ACMDS validation
    print("\nValidating ACMDS components:")
    all_good &= validate_acmds()

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✓ All core components validated successfully!")
        print("\nYou can now use ACMDS optimizer:")
        print("  - Python API: import acmds_optimizer")
        print("  - CLI: python train.py --help")
    else:
        print("✗ Some components failed validation")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
