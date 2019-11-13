# import torch
#
# l = torch.tensor(5)
#
# print(l)
import numpy as np
import torch

# pboxes = [[231.55981553,  87.82314461, 258.5250043,  115.4244695,    0.60790879], [229.28632851, 230.35264409, 255.48803055, 258.57825062, 0.60135394]]
# pboxes = np.array(pboxes)
# print(pboxes[:])
# print(pboxes[:, 0])


# a = np.array([1, 2, 3])
# b = np.array([2, 3, 4])
# print(a.shape)
# print(b.shape)
# print(np.stack((a, b), axis=0))
# print(np.stack((a, b), axis=0).shape)
# print(np.stack((a, b), axis=1))
# print(np.stack((a, b), axis=1).shape)
# a = np.array([[[1, 2, 3],  [1, 2, 3]], [[1, 2, 3],  [1, 2, 3]]])
# a = torch.tensor(a)
# # print(a)
# # print(a.shape)
# # print(np.stack(a, axis=0), "np0")
# print(torch.stack(a, 0), "torch0")
# print(np.stack(a, axis=1), "np1")
# print(torch.stack(a, dim=1), "torch1")
# print(np.stack(a, axis=2), "np2")
# print(torch.stack(a, dim=2), "torch2")
# print(b)
# print(b.shape)
# print(np.stack((a, b), axis=0))
# print(np.stack((a, b), axis=0).shape)
# print(np.stack((a, b), axis=1))
# print(np.stack((a, b), axis=1).shape)
# print(np.stack((a, b), axis=2))
# print(np.stack((a, b), axis=2).shape)
# a = torch.ones([1, 2, 3])
# print(type(a))
# b = torch.ones([1, 2, 3])
# print(b)
# print(torch.stack([a, b], 0))
# print(torch.stack([a, b], 0).shape)
# print(torch.stack([a, b], 1))
# print(torch.stack([a, b], 1).shape)
a = [1,2]
b = [3,4]
print(a+b)