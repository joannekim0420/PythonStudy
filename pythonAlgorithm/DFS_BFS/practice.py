#Depth First Search

def dfs(graph, v, visited):
    visited[v] = True
    print(v, end='')
    for i in graph[v]:
        if not visited[i]:
            dfs(graph,i,visited)

visited = [False]*9
graph =[
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

from collections import deque
#Breadth First Search
def bfs(graph, start, visited):
    queue = deque([start])
    visited[start] = True
    while queue:
        v = queue.popleft()
        print(v, end='')
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
bfs(Graph, 1, visited)