n, m = map(int, input().split())
ball = list(map(int, input().split()))

count, i = 0, 0
for x in ball:
    i+=1
    for y in ball[i:n+1]:
        if x != y:
          count +=1

print(count)

# array = [0]*11
#
# for i in ball:
#     array[i] += 1
# result = 0
# for i in range(1, m+1):
#     n -= array[i]
#     result += array[i] *n
# print(result)



