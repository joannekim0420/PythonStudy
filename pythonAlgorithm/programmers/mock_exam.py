import sys

input = sys.stdin.readlines()


def solution(answers):
    answer = []
    result = []
    ppl = [[1, 2, 3, 4, 5] * len(answers), [2, 1, 2, 3, 2, 4, 2, 5] * len(answers),
           [3, 3, 1, 1, 2, 2, 4, 4, 5, 5] * len(answers)]

    for i in range(len(ppl)):
        count = 0
        for j in range(len(answers)):
            if ppl[i][j] == answers[j]:
                count += 1

        result.append((i + 1, count))
        result = sorted(result, key=lambda x: x[1], reverse=True)

    max = -1
    for i in range(len(ppl)):
        if result[i][1] >= max:
            max = result[i][1]
            answer.append(result[i][0])

    return answer