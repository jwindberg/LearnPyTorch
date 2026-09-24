"""
Module 1: PyTorch Tensors - The Building Blocks of Deep Learning

In this script, you will learn:
1. Creating tensors from Python lists and built-in functions
2. Inspecting tensor properties: shape, dtype, and device
3. Tensor arithmetic and broadcasting
4. Reshaping and slicing
5. Moving tensors to the GPU / MPS device
"""

import torch

def section(title: str):
    print("\n" + "=" * 50)
    print(f"📘 {title}")
    print("=" * 50)

def main():
    # -------------------------------------------------------------
    # 1. Creating Tensors
    # -------------------------------------------------------------
    section("1. Tensor Creation")
    # From standard Python lists
    scalar = torch.tensor(7)
    vector = torch.tensor([1, 2, 3])
    matrix = torch.tensor([[1, 2], [3, 4]])
    
    print("Scalar (0-D):", scalar, "| ndim:", scalar.ndim)
    print("Vector (1-D):", vector, "| shape:", vector.shape)
    print("Matrix (2-D):\n", matrix, "| shape:", matrix.shape)

    # Built-in generator functions
    zeros = torch.zeros(2, 3)                # 2 rows, 3 cols of zeros
    ones = torch.ones(2, 3)                  # 2 rows, 3 cols of ones
    rand = torch.randn(2, 3)                 # Standard normal distribution (mean=0, std=1)
    arange = torch.arange(0, 10, step=2)     # [0, 2, 4, 6, 8]
    
    print("\nRandom (2x3):\n", rand)
    print("Arange [0, 10, step 2]:", arange)

    # -------------------------------------------------------------
    # 2. Tensor Attributes (Shape, Dtype, Device)
    # -------------------------------------------------------------
    section("2. Essential Tensor Attributes")
    sample = torch.rand(3, 4, dtype=torch.float32)
    print(f"Shape:  {sample.shape}")
    print(f"Dtype:  {sample.dtype}")
    print(f"Device: {sample.device}")

    # -------------------------------------------------------------
    # 3. Basic Operations & Broadcasting
    # -------------------------------------------------------------
    section("3. Math & Broadcasting")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("a + b (Element-wise):", a + b)
    print("a * b (Element-wise):", a * b)
    print("Dot product (torch.dot):", torch.dot(a, b))

    # Matrix multiplication: (2x3) @ (3x2) -> (2x2)
    m1 = torch.randn(2, 3)
    m2 = torch.randn(3, 2)
    mm_result = m1 @ m2  # Or torch.matmul(m1, m2)
    print("\nMatrix multiplication shape (m1 @ m2):", mm_result.shape)

    # -------------------------------------------------------------
    # 4. Reshaping & Slicing
    # -------------------------------------------------------------
    section("4. Reshaping & Slicing")
    x = torch.arange(1, 10)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("Original 1D tensor:", x)

    # Reshaping into 3x3 matrix
    x_3x3 = x.view(3, 3)
    print("Reshaped into 3x3:\n", x_3x3)

    # Slicing: first two rows, last column
    slice_out = x_3x3[:2, -1]
    print("First 2 rows, last column:", slice_out)

    # Unsqueeze (add a batch/channel dimension): (3, 3) -> (1, 3, 3)
    expanded = x_3x3.unsqueeze(0)
    print("Unsqueeze dim 0 shape:", expanded.shape)

    # -------------------------------------------------------------
    # 5. Device Management (CPU vs MPS/GPU)
    # -------------------------------------------------------------
    section("5. Device Transfers")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute target: {device}")

    cpu_tensor = torch.ones(2, 2)
    gpu_tensor = cpu_tensor.to(device)
    print("Device of cpu_tensor:", cpu_tensor.device)
    print("Device of gpu_tensor:", gpu_tensor.device)

    print("\n🎉 Module 1 complete! You know how tensors work.")

if __name__ == "__main__":
    main()
