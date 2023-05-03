import torch.nn as nn
import torch
import torch.nn.functional as F

batch_size = 2
one_hot_size = 10
sequence_width = 7
data = torch.randn(batch_size, one_hot_size, sequence_width)
conv1 = nn.Conv1d(in_channels=one_hot_size,out_channels=16, kernel_size = 3)

layer = conv1(data)

print(layer)
print(data.size())
print(layer.size())

