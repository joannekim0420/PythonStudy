import heapq
import sys
input = sys.stdin.readline

INF = int(1e9)

for tc in range(int(input())):
    n = int(input())
    graph = []
    for _ in range(n):
        graph.append(list(map(int,input().split())))

    distance = [[INF]*(n) for _ in range(n)] ##원래 1row의 list 가 아닌 2d로

    q = []
    x, y = 0, 0
    distance[x][y] = graph[x][y] #시작 노드 초기화
    heapq.heappush(q, (graph[x][y],x,y))
    while q:
        dist, x, y = heapq.heappop(q)
        dx = [0,0,-1,1]
        dy = [-1,1,0,0]
        if distance[x][y] < dist:
            continue
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            if nx>=0 and nx<n and ny>= 0 and ny<n:
                new_dist = dist+graph[nx][ny]
                if new_dist<distance[nx][ny]:
                    distance[nx][ny] = new_dist
                    heapq.heappush(q,(new_dist,nx,ny))
    print(distance[n-1][n-1])