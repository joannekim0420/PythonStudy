    from collections import deque

n, m = map(int,input().split())
graph, virus = [], []
tmp = [[0]*m for _ in range(n)]

for i in range(n):
    graph.append(list(map(int,input().split())))
    for j in range(m):
        if graph[i][j] == 2:
            virus.append((i,j))

dx = [0, 0, -1, 1]
dy = [-1, 1, 0 ,0]

max_safezone = -1e9
def spread_virus():
    q = deque(virus)
    count_zero = 0

    for i in range(n):
        for j in range(m):
            tmp[i][j] = graph[i][j]

    while q:
        x, y = q.popleft()
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            if nx>=0 and nx<n and ny>=0 and ny<m and tmp[nx][ny] == 0:
                tmp[nx][ny] = 2
                q.append((nx,ny))

    for i in range(n):
        for j in range(m):
            if tmp[i][j] == 0:
                count_zero += 1
    return count_zero

def dfs(count):
    global max_safezone

    if count == 3:
        max_safezone = max(max_safezone, spread_virus())
        return
    else:
        for i in range(n):
            for j in range(m):
                if graph[i][j] == 0:
                    graph[i][j] = 2
                    count +=1
                    dfs(count)
                    graph[i][j] = 0
                    count-=1
                else:
                    continue

dfs(0)
print(max_safezone)