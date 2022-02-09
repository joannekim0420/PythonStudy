import torch

#calculation
x = torch.randn(2,3)
print(x+x)
print(torch.add(x,x))

#dimension
x = torch.arange(6)
print(x)
x = x.view(2,3)
print(x)

print(torch.sum(x, dim=0)) #2*3 을 dim=0 행에 대해 합침
print(torch.sum(x, dim=1)) #2*3 을 dim=1 열에 대해 합침

#transpose
print(torch.transpose(x,0,1))

#indexing. slicing, concat
print()
x = torch.arange(6).view(2,3)
print(x)
print(x[:1,:2]) #0행1열까지
print(x[0,1]) #0행1열의 값

print()
print(torch.cat([x,x], dim=0))
print(torch.cat([x,x], dim=1))
print(torch.stack([x,x]))

x2= torch.ones(3,2)
print(x2)
x2[:,1]+=1 #모든 행의 1열의 값을 +1
print(x2)

print()
x1 = torch.FloatTensor([[0,1,2],[3,4,5]])
print(x1)
print(torch.mm(x1,x2))
