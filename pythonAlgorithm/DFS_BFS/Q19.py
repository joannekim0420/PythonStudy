from itertools import permutations
import sys

input = sys.stdin.readlines()

n = int(input())

number = list(map(int,input().split()))

add, sub, mul, div = map(int,input().split())
operator = []
for _ in range(n-1):
    if add>0:
        operator.append('add')
        add -=1
    if sub >0:
        operator.append('sub')
        sub -= 1
    if mul >0 :
        operator.append('mul')
        mul -= 1
    if div > 0:
        operator.append('div')
        div -= 1
# print(operator)
permute = list(permutations(operator, n-1))

max_value = -1e9
min_value = 1e9

for p in permute:
    result = number[0]
    for i in range(n - 1):
        if p[i] == 'add':
            result = result + number[i+1]
        elif p[i] == 'sub':
            result = result - number[i+1]
        elif p[i] == 'mul':
            result = result * number[i+1]
        elif p[i] == 'div':
            result = int(result / number[i+1])
        else:
            pass
    max_value = max(max_value, result)
    min_value = min(min_value, result)

print(max_value)
print(min_value)

