import torch

x = torch.randn(2,3,5)
print("x",x)
print("x.size():",x.size())
print("x.shape():",x.shape)

print("x permute",x.permute(2,0,1))
print(torch.permute(x, (2,0,1)).size())

