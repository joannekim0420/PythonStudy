import heapq
# import sys
# input = sys.stdin.read
# n = int(input())
#
# #heap
# heap = []
# for i in range(n):
#     data = int(input())
#     heapq.heappush(heap, data)
#
# result = 0
# while len(heap) != 1:
#     one = heapq.heappop(heap)
#     two = heapq.heappop(heap)
#
#     sum = one + two
#     result += sum
#     heapq.heappush(heap, sum)
# print(result)


n = int(input())
cards = []

for i in range(n):
    data = int(input())
    heapq.heappush(cards, data)

print(cards)
sum_all = 0
while len(cards) != 1:
    one = heapq.heappop(cards)
    two = heapq.heappop(cards)
    sum_cand = one + two
    heapq.heappush(cards, sum_cand)
    sum_all += sum_cand

print(sum_all)