import torch
print("version:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_ver:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
