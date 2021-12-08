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
#리스트로 안 됨!
#예외
# [30,40,50,60,90]
