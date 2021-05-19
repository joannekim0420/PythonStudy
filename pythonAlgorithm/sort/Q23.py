n = int(input())
data = []
for i in range(n):
    data.append(tuple(input().split()))

sort = sorted(data, key = lambda x: (-int(x[1]),int(x[2]), -int(x[3]), x[0])) #reverse = True 내림차순

for name in sort:
    print(name[0])

