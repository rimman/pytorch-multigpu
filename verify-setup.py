from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)


print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('Active CUDA Device: GPU', torch.cuda.current_device())
print('CUDA Device Name:', torch.cuda.get_device_name(0))

