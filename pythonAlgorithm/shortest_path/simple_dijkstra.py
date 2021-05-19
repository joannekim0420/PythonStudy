import sys
input = sys.stdin.readline
INF = int(1e9)

#n = 노드의 개수, m = 간선의 개수
n, m = map(int, input().split())

start = int(input())
graph = [[] for i in range(n+1)]
visited = [False]*(n+1)
distance = [INF]*(n+1)

for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a].append((b,c)) #a노드에서 b노드로 가는 비용 c

def get_smallest_node(): #방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
    min_value = INF
    index =0
    for i in range(1, n+1):
        if distance[i] < min_value and not visited[i]:
            min_value = distance[i]
            index = i
    return index

def dijkstra(start):
    #시작노드에서 초기화
    distance[start] = 0
    visited[start] = True
    for j in graph[start]:
        distance[j[0]] = j[1]
    #전체 노드 개수만큼 해주는데, 마지막 노드는 어차피 방문해봤자라 -1
    for i in range(n-1):
        now = get_smallest_node()
        visited[now] = True
        #현재 노드와 연결된 다른 노드 확인하여 최단 거리 비교 갱신
        for j in graph[now]:
            cost = distance[now]+j[1]
            if cost < distance[j[0]]:
                distance[j[0]] = cost

dijkstra(start)
print(graph)
print(graph[start])
for i in range(1,n+1):
    if distance[i] == INF:
        print("INF")
    else:
        print("shortest distance from[",start,"] to [",i,"] is ",distance[i])
