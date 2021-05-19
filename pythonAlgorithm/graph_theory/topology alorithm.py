from collections import deque

v, e = map(int, input().split())
indegree = [0]*(v+1)
graph = [[] for i in range(v+1)]

for _ in range(e):
    a, b = map(int, input().split())
    # a to b
    graph[a].append(b)
    # b로 진입하는 간선의 개수 indegree 증가
    indegree[b] +=1

def topology_sort():
    result = []
    q = deque()

    #insert into queue node which indegree is 0
    for i in range(1, v+1):
        if indegree[i] == 0:
            q.append(i)

    while q:
        now = q.popleft()
        result.append(now)
        # 해당 원소와 연결 된 노드들의 진입차수에서 1 빼기
        for i in graph[now]:
            indegree[i] -=1
            #새롭게 진입차수가 0이 되는 노드를 큐에 삽입
            if indegree[i] == 0:
                q.append(i)
    for i in result:
        print(i, end=' ')

topology_sort()