def find_one(x):
    count = 0
    while x != 1:
        if x % 5 ==0:
            x //= 5
            count += 1
        elif x % 3 == 0:
            x //= 3
            count += 1
        elif x % 2 == 0:
            x //= 2
            count += 1
        else:
            x -= 1
            count += 1
    return count

x = int(input())

d=[0]*30001

for i in range(2, x+1):
    d[i] = d[i-1] + 1
    if i%2 ==0:
        d[i] = min(d[i], d[i//2]+1)
    if i%3 == 0:
        d[i] = min(d[i], d[i // 3] + 1)
    if i%5 == 0:
        d[i] = min(d[i], d[i // 5] + 1)

print(d[x])