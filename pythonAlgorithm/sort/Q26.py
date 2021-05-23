import heapq
import sys
input = sys.stdin.read
n = int(input())

#heap
heap = []
for i in range(n):
    data = int(input())
    heapq.heappush(heap, data)

result = 0
while len(heap) != 1:
    one = heapq.heappop(heap)
    two = heapq.heappop(heap)

    sum = one + two
    result += sum
    heapq.heappush(heap, sum)
print(result)

#list#
# cards = []
# for i in range(n):
#     cards.append(int(input()))
#
# cards.sort()
# sum = 0
# for i in range(len(cards)):
#     if i==0:
#         a = sum+cards[i]
#     else:
#         a = a+cards[i]
#         sum += a
# print(sum)
