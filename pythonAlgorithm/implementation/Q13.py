from itertools import combinations

n,m = map(int, input().split())
# graph = [[0,0,1,0,0],[0,0,2,0,1],[0,1,2,0,0],[0,0,1,0,0],[0,0,0,0,2]]
# graph = [[0,2,0,1,0],[1,0,1,0,0],[0,0,0,0,0],[2,0,0,1,1],[2,2,0,1,2]]
graph = []
house,chicken = [], []

for _ in range(n):
    graph.append(list(map(int, input().split())))

for i in range(n):
    for j in range(n):
        if graph[i][j] == 1:
            house.append((i+1,j+1))
        elif graph[i][j] == 2:
            chicken.append((i+1,j+1))
        else:
            pass

candidates = list(combinations(chicken,m)) #all combinations

def get_sum(candidate):
    dist, sum = 1e9,0
    for hx, hy in house:
        for cx, cy in candidate:
            diff = abs(hx-cx) + abs(hy-cy)
            if diff<dist:
                dist = diff
            # dist = min(dist, abs(hx-cx) + abs(hy-cy))
        sum += dist
    return sum
result = 1e9

for candidate in candidates:
    result = min(result, get_sum(candidate))

print(result)