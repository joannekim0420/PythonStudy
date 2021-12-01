from collections import deque

n, k = map(int, input().split())

graph, virus = [], []

for i in range(n):
    graph.append(list(map(int, input().split())))
    for j in range(n):
        if graph[i][j] != 0:
            virus.append((graph[i][j], i, j))

dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

S, X ,Y = map(int, input().split())

def spread_virus(s, X, Y):
    time = 0
    q = deque(virus)
    while q:
        if time == s:
            break

        for _ in range(len(q)):
            virus_num, a, b = q.popleft()
            for i in range(4):
                nx = a + dx[i]
                ny = b + dy[i]

                if nx >= 0 and nx<n and ny>=0 and ny<n and graph[a][b]!=0:
                    graph[nx][ny] = graph[a][b]
                    q.append((graph[nx][ny],a,b))
                else: continue
        time +=1
    return graph[X-1][Y-1]

virus.sort()
print(spread_virus(S, X, Y))