n = int(input())

member = list(map(int, input().split()))

member.sort()

result = 0
count =0

for i in member:
    count += 1
    if count >= i:
        result +=1
        count =0

print(result)