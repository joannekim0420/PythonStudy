
def rotate_90_clockwise(matrix):
    n = len(matrix)
    m = len(matrix[0])
    result_matrix = [[0]*n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            result_matrix[j][(n-1)-i] = matrix[i][j]

    return result_matrix

key = [[0,0,0],[1,0,0],[0,1,1]]
lock = [[1,1,1],[1,1,0],[1,0,1]]

len_lock = len(lock)
try_lock = [[0] * (len_lock*3) for _ in range(len_lock*3)]
for i in range(len_lock):
    for j in range(len_lock):
        try_lock[i+len_lock][j+len_lock] = lock[i][j]

def check(tmp):
    length = len(tmp) // 3
    for i in range(length, length*2):
        for j in range(length, length*2):
            if tmp[i][j] != 1:
                return False
    return True


def solution(key, lock):
    for i in range(len_lock*2):
        for j in range(len_lock*2):
            for _ in range(4):
                tmp_lock = [[0] * (len_lock) for _ in range(len_lock)]
                for n in range(len(key)):
                    for m in range(len(key)):
                        try_lock[i+n][j+m] += key[n][m]

                if check(try_lock) == True:
                    return True
                else:
                    for n in range(len(key)):
                        for m in range(len(key)):
                            try_lock[i+n][j+m] -= key[n][m]

                key = rotate_90_clockwise(key)

    return False

print(solution(key,lock))