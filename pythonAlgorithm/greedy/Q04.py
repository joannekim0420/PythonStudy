n = int(input())
money = list(map(int, input().split()))
money.sort()

min = 1
for x in money:
    if min < x:
        break
    min += x

print(min)