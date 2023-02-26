import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("TRUE")
    print("Count: ",torch.cuda.device_count())
    print("Memory: ", torch.cuda.memory_allocated())
    print("Memory Cache: ", torch.cuda.memory_reserved())
else:
    device = torch.device("cpu")

import torch

# Get the amount of GPU memory currently in use
memory_used = torch.cuda.memory_allocated() / 1024 ** 2
print(f"GPU memory used: {memory_used:.2f} MB")

# Get the maximum amount of GPU memory that has been reserved by PyTorch
memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
print(f"GPU memory reserved: {memory_reserved:.2f} MB")
