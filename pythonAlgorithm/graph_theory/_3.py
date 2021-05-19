#백준 도시분할계획 1647
import sys
input = sys.stdin.readline

def find_parent(parent, x):
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

n,m = map(int, input().split())
parent = [0]*(n+1)
result = 0
graph = []
for i in range(n+1):
    parent[i] = i
for i in range(m):
    a, b, cost = map(int, input().split())
    graph.append((cost, a, b))

graph.sort()
last = 0
for cur in graph:
    cost, a, b = cur
    if find_parent(parent, a) != find_parent(parent, b):
        union_parent(parent, a, b)
        result += cost
        last = cost

print(result-last)


