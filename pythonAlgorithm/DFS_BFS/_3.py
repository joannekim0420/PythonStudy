n,m = map(int, input().split())
graph = []
for i in range(n):
    graph.append(list(map(int,input())))    #맵 정보 받기 2차원으로

def dfs(x,y):
    if x<=-1 or x>=n or y<=-1 or y>=m:      #2차원을 벗어나는 구역은 False return
        return False

    if graph[x][y]==0:                      #해당 지점이 0, 즉 방문할 수 있는 지역일 때
        graph[x][y]=1                       #해당 값을 다시 반복해서 방문하지 않도록 1로 바꿔줌
        dfs(x+1,y)                          #상하좌우 연결된 지점 recursive 찾아서 모두 1로 바꿔줌
        dfs(x-1,y)                          #즉, 상하좌우 0으로 연결되면 하나의 지역으로 간주
        dfs(x,y+1)
        dfs(x,y-1)
        return True                         #if dfs(i,j)==True 조건일때만 result 값 증가
    return False
result =0
for i in range(n):
    for j in range(m):
        if dfs(i,j)==True:
            result += 1

print(result)