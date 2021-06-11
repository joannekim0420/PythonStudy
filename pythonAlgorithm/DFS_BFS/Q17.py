from collections import deque

n, k = map(int, input().split())
data = []

for _ in range(n):
    data.append(list(map(int, input().split())))

s,X,Y = map(int, input().split())

# print(data)
dx = [-1,1,0,0]
dy = [0,0,-1,1]

# def dfs(x,y):
#     if x<=-1 or x>=n or y<=-1 or y>=n:
#         return False
#     if data[x][y] == 1:
#
def bfs(x,y, s):
    queue = deque()
    queue.append((x,y))
    while s>0:
        x, y = queue.popleft()
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            if nx <0 or ny<0 or nx>=n or ny>=n:
                continue
            if data[nx][ny] == 0:
                if data[x][y] == 1:
                    data[nx][ny] = 1
                elif data[x][y] == 2:
                    data[nx][ny] =2
                elif data[x][y] == 3:
                    data[nx][ny] = 3
                else:
                    continue
                queue.append((nx,ny))
            else:
                continue
        s-=1
    return data[X-1][Y-1]
print(bfs(0,0,s))

