import heapq

def solution(food_tiems, k):
    if sum(food_tiems) <= k:
        return -1
    q = []
    for i in range(len(food_times)):
        heapq.heappush(q,(food_tiems[i], i+1))

    sum_value =0
    previous = 0
    length = len(food_times)

    while sum_value +((q[0][0] - previous)*length)<=k:
        now = heapq.heappop(q)[0]
        print(now)
        sum_value += (now-previous)*length
        length -= 1
        previous = now

    result = sorted(q, key=lambda x:x[1])
    return result[(k-sum_value) % length][1]

food_times = [3,2,2]
print(solution(food_times, 5))