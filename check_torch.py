try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
except ImportError:
    print("Torch not installed")
