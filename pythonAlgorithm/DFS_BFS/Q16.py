from itertools import combinations
from collections import deque

n,m = map(int, input().split())
graph = []

for _ in range(n):
    graph.append(list(map(int,input().split())))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(x,y):
    queue = deque()