import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Test by creating a tensor on the GPU
    x = torch.rand((3, 3), device=device)
    print("Tensor on GPU:", x)
else:
    print("GPU not available, running on CPU.")
    device = torch.device("cpu")
