from collections import deque

n = int(input())
graph, teacher = [], []

for i in range(n):
    graph.append(list(map(int, input().split())))
    for j in range(n):
        if graph[i][j] == 'T':
            teacher.append((i,j))

def dfs(count):
    if count == 3:
        if look_out():
            return True
        else:
            return

    for i in range(n):
        for j in range(n):
            if graph[i][j] == 'X':
                graph[i][j] = 'O'
                count +=1
                dfs(count)
                graph[i][j] = 'X'
                count -=1

dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

def look_out():
    q = deque(teacher)

    while q:
        x, y = q.popleeft()
        for i in range(4):  # 0:L / 1:R / 2:U / 3:D
            nx = x + dx[i]
            ny = y + dy[i]
            if nx>=0 and nx<n and ny>=0 and ny<n and graph[nx][ny] == 'X':
                check(nx, ny, i)

def check(x,y,i):
    nx = x + dx[i]
    ny = y + dy[i]

    if graph[x][y] == "X":
        check(nx, ny, i)
    elif graph[x][y] == "S":
        return False
    else:
        return True



    if i == 0: #Left
        nx = x + dx[i]
    elif i == 1: #Right

    elif i == 2: #Up

    else: #Down

