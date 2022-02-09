import torch
import numpy as np

def describe(x):
    print("type: {}".format(x.type()))
    print("shape(크기): {}".format(x.shape))
    print("값: {}".format(x))

describe(torch.Tensor(2,3))
describe(torch.rand(2,3)) #균등 분포
describe(torch.randn(2,3)) #표준 정규 분포

print()

describe(torch.zeros(2,3))
describe(torch.ones(2,3))
x = torch.ones(2,3)
x.fill_(5)
describe(x)

#list to torch
describe(torch.Tensor([[1,2,3],[4,5,6]]))

#numpy to torch
npy = np.random.rand(2,3)
describe(torch.from_numpy(npy))

