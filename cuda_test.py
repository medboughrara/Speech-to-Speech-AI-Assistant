import torch
import os

print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Check cuDNN
if torch.backends.cudnn.is_available():
    print("cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN is enabled:", torch.backends.cudnn.enabled)
else:
    print("cuDNN is not available")

# Print relevant environment variables
cuda_path = os.environ.get('CUDA_PATH')
cuda_home = os.environ.get('CUDA_HOME')
print("\nEnvironment Variables:")
print("CUDA_PATH:", cuda_path)
print("CUDA_HOME:", cuda_home)