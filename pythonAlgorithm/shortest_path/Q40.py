import sys
import heapq

INF = int(1e9)
input = sys.stdin.readline

n ,m = map(int, input().split())
graph = [[] for _ in range(n+1)]
for i in range(m):
    a,b = map(int, input().split())
    graph[a].append((b,1))
    graph[b].append((a,1))#to_node, weight

distance = [INF]*(n+1)

q = []
heapq.heappush(q, (0,1))
distance[1] = 0
while q:
    dist, now = heapq.heappop(q)
    if distance[now] < dist:
        continue
    for i in graph[now]:
        next = dist+i[1]
        if next < distance[i[0]]:
            distance[i[0]] = next
            heapq.heappush(q,(next,i[0]))

count, min_idx = 0, -1
for i in range(n+1):
    if distance[i] == INF:
        distance[i] = 0
print(distance)
max_len = max(distance)
for i in range(1, n+1):
    if max_len == distance[i]:
        if min_idx == -1:
            min_idx = i
            print(min_idx, end= ' ')
        count+=1
print(max_len, end=' ')
print(count)

