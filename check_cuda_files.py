import os

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
required_files = [
    "cublas64_12.dll",
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn_infer64_9.dll",
    "cudnn_adv_infer64_9.dll"
]

print("Checking CUDA files in:", cuda_path)
print("\nRequired files status:")
found_files = os.listdir(cuda_path) if os.path.exists(cuda_path) else []
for file in required_files:
    status = "✓ Found" if file in found_files else "✗ Missing"
    print(f"{file}: {status}")