import numpy as np
import torch


a = np.random.random((1, 4,4,3))
a = torch.from_numpy(a)
b = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
b = torch.from_numpy(b)
print(b.shape)
print(b)
print(a)

a[:,:,:,2]=b
print(a)