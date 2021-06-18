from collections import deque
def solution(priorities, location):
    answer = 0
    queue = deque(priorities)
    m = max(queue)
    print(m)
    while True:
        v = queue.popleft()
        if m == v:
            answer += 1
            if location == 0:
                break
            else:
                location -= 1
            m = max(queue)
        else:
            queue.append(v)
            if location == 0:
                location = len(queue) - 1
            else:
                location -= 1
    return answer
