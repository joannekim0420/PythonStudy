from collections import deque
# def solution(priorities, location):
#     answer = 0
#     queue = deque(priorities)
#     m = max(queue)
#     print(m)
#     while True:
#         v = queue.popleft()
#         if m == v:
#             answer += 1
#             if location == 0:
#                 break
#             else:
#                 location -= 1
#             m = max(queue)
#         else:
#             queue.append(v)
#             if location == 0:
#                 location = len(queue) - 1
#             else:
#                 location -= 1
#     return answer
#

#####2번째 풀이
# def solution(priorities, location):
#     answer = 0
#     queue = deque(priorities)
#     while queue:
#         print(queue)
#         now = queue.popleft()
#         if location == 0:
#             if now >= max(queue):  # 내가 원하는게 우선 순위가 제일 높고 프린트 순서일 때
#                 break
#             else:  # 내가 원하는게 프린트 순서이지만 우선 순위가 제일 높지 않을 때
#                 queue.append(now)
#                 location = len(queue) - 1
#         else:  # 내가 원하는게 나오지 않았을 때
#             if now >= max(queue):  # 프린트 순서이면서 가장 우선순위가 높을 때
#                 answer += 1
#             else:  # 내가 원하는게 나오지 않았지만 가장 우선순위가 높지 않을 때
#                 queue.append(now)
#             location -= 1
#
#     return answer



def solution(priorities, location):
    answer = 0
    search, c = sorted(priorities, reverse=True), 0
    while True:
        for i, priority in enumerate(priorities):
            s = search[c]
            if priority == s:
                c += 1
                answer += 1
                if i == location:
                    break
        else:
            continue
        break
    return answer

# priorities = [2, 1, 3, 2]
# location = 2

# priorities = [1, 1, 9, 1, 1, 1]
# location = 0

priorities = [1, 2, 3]
location = 0

print(solution(priorities, location))