def solution(land):
    d = [[0]*4 for _ in range(len(land)+1)]
    for i in range(len(land)):
        for j in range(4):
            for col in range(4):
                if j == col:
                    continue
                d[i+1][j] = max(d[i+1][j],land[i][j]+d[i][col])
    print(d)
#land = [[1, 2, 3, 5], [5, 6, 7, 8], [4, 3, 2, 1], [3, 4, 5, 6]]
land = [[1, 2, 3, 5], [5, 6, 7, 8], [4, 3, 2, 1]]
solution(land)