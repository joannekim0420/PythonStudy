from collections import deque

n, m, k, x = map(int, input().split())

#graph = [[]*(n+1)]
graph = [[],[2,3],[3,4],[],[]]

# for _ in range(m):
#     a, b = map(int, input().split())
#     graph[a].append(b)

distance = [-1]*(n+1)
distance[x] = 0

q = deque([x])

while q:
    now = q.popleft()
    for next in graph[now]:
        if distance[next] == -1:
            q.append(next)
            distance[next] = distance[now]+1

check = False
for i in range(1,n+1):
    if distance[i] == k:
        print(i)
        check = True

if check == False:
    print(-1)
