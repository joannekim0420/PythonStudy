d = [0]*101

x = int(input())
data = list(map(int,input().split()))
d[0],d[1] = data[0],data[1]

for i in range(2,len(data)):
    d[i] = max(d[i-2]+data[i],d[i-1])

print(d[x-1])