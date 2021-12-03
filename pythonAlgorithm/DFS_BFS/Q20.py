n = int(input())
graph, teacher = [], []
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

for i in range(n):
    graph.append(list(map(str, input().split())))
    for j in range(n):
        if graph[i][j] == 'T':
            teacher.append((i,j))

def dfs(count):
    if count == 3:
        if look_out():
            return True
        else:
            return False
    else:
        for i in range(n):
            for j in range(n):
                if graph[i][j] == 'X':
                    graph[i][j] = 'O'
                    count +=1
                    if dfs(count) == False:
                        graph[i][j] = 'X'
                        count -=1
                        continue
                    else:
                        return True
    return False

def look_out():
    for q in teacher:
        x,y = q
        for i in range(4):  # 0:L / 1:R / 2:U / 3:D
            nx = x + dx[i]
            ny = y + dy[i]
            if nx>=0 and nx<n and ny>=0 and ny<n:
                if graph[nx][ny] == 'X':
                    if check_line(nx, ny, i):
                        continue
                    else:
                        return False
                elif graph[nx][ny] == 'S':
                    return False
                else:
                    continue
    return True

def check_line(x,y,i):
    nx = x + dx[i]
    ny = y + dy[i]
    if nx>=0 and nx<n and ny>=0 and ny<n:
        if graph[nx][ny] == "X":
            if check_line(nx, ny, i):
                return True
            else:
                return False
        elif graph[nx][ny] == "S":
            return False
        else:
            return True
    return True

if dfs(0): print("YES")
else: print("NO")
