import sys
input = sys.stdin.readlines()
INF = int(1e9)

n,m,k,x = map(int,input().split())

graph = [[] for i in range(n+1)]
distance = [INF]*(n+1)
visited = [False]*(n+1)

def get_smallest_node():
    min_value = INF
    index = 0
    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index

for _ in range(m):
    # graph.append(list(map(int,input())))
    a, b = map(int, input().split())
    graph[a].append(b)

def dijkstra(x):
    distance[x] = 0
    visited[x] = True
    for j in graph[x]:
        distance[j] = 1

    for i in range(n-1):
        now = get_smallest_node()
        visited[now] = True
        for j in graph[now]:
            cost = distance[now]+1
            if cost<distance[j]:
                distance[j]=cost
dijkstra(x)

check = False
for i in range(n+1):
    if distance[i] == k:
        print(i)
        check = True

if check == False:
    print(-1)

