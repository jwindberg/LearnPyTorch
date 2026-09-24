"""
Module 0: Verify PyTorch & Hardware Setup
Checks PyTorch installation, version, and hardware acceleration (MPS for Apple Silicon or CUDA).
"""

import sys
import torch

def verify_environment():
    print("=" * 50)
    print("🔍 PyTorch Environment Verification")
    print("=" * 50)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check for Apple Silicon GPU acceleration (Metal Performance Shaders)
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    # Check for CUDA (NVIDIA GPUs)
    cuda_available = torch.cuda.is_available()
    
    print(f"Apple Silicon MPS available: {mps_available} (Built: {mps_built})")
    print(f"CUDA available: {cuda_available}")
    
    if mps_available:
        device = torch.device("mps")
        print("🚀 Primary Acceleration Device: Apple Metal (MPS)")
    elif cuda_available:
        device = torch.device("cuda")
        print(f"🚀 Primary Acceleration Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("💻 Primary Acceleration Device: CPU")

    # Quick test tensor allocation on the target device
    print("\nRunning a quick tensor test on", device, "...")
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = torch.matmul(x, y)
    print("Test calculation output:")
    print(z)
    print("\n✅ Setup verified successfully! You are ready for Lesson 1.")
    print("=" * 50)

if __name__ == "__main__":
    verify_environment()
