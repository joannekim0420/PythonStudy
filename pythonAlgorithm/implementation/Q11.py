
n = int(input())
k = int(input())
data = [[0]*(n+1) for _ in range(n+1)]

for _ in range(k):
    x, y = map(int, input().split())
    data[x][y] = 1

l = int(input())
move = []
for _ in range(l):
    x, c = input().split()
    move.append((int(x),c))

dx = [0 , 1, 0, -1]
dy = [1, 0 , -1, 0]

def turn(c, direction):
    if c == 'L':
        direction = (direction-1)%4
    else:
        direction = (direction+1)%4

    return direction

def go_snake():
    direction = 0
    index = 0
    x, y = 1, 1
    time = 0
    q = [(x,y)] #snake head,body,tail
    data[x][y] = 2 #2 equals snake, 1 equals apple, 0 equals empty space

    while True:
        nx = dx[direction]+x #future step
        ny = dy[direction]+y
        if 1<=nx and nx<=n and 1<=ny and ny<=n and data[x][y] !=2:
            if data[nx][ny] == 0:
                data[nx][ny] = 2
                q.append((nx, ny))
                px, py = q.pop(0) #past snake tail
                data[px][py] = 0

            if data[nx][ny] == 1: # apple
                data[nx][ny] = 2
                q.append((nx,ny))

        else:
            time +=1
            break
        time +=1
        if index<l and time == move[index][0]: #회전수가 아직 남아있고, 회전 시간이 됐을 때
            direction = turn(move[index][1],direction)
            index +=1
    return time

print(go_snake)



"""n = int(input())
k = int(input())

data = [[0]*(n+1) for _ in range(n+1)]
info = []
for _ in range(k):
    a, b = map(int, input().split())
    data[a][b] = 1

l = int(input())
for _ in range(l):
    x, c = input().split()
    info.append((int(x), c))

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

def turn(direction, c):
    if c == "L":
        direction = (direction - 1) % 4
    else:
        direction = (direction + 1) % 4
    return direction

def simulate():
    x, y = 1, 1
    data[x][y] = 2
    direction = 0
    time = 0
    index = 0
    q = [(x,y)]
    while True:
        nx = x+dx[direction]
        ny = y+dy[direction]

        if 1<=nx and nx<= n and 1<= ny and ny<=n and data[nx][ny] !=2:
            if data[nx][ny] == 0:
                data[nx][ny] = 2
                q.append((nx,ny))
                px, py = q.pop(0)
                data[px][py] = 0

            if data[nx][ny] == 1:
                data[nx][ny] = 2
                q.append((nx,ny))

        else:
            time+=1
            break
        x,y = nx, ny
        time +=1
        if index < l and time == info[index][0]:
            direction = turn(direction, info[index][1])
            index +=1

    return time

print(simulate())"""
