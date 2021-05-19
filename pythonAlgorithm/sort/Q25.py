def solution(N, stages):
    answer = []
    fail_rate = []
    count = [0 for i in range(N+2)]
    stages.sort()
    #count를 for문 하나로 쉽게 할 수 있는 방법은 stages.count(c)
    # for i in stages:
    #     count[i] += 1

    length = len(stages)

    for c in range(1, N+1):
        s_count = stages.count(c)
        if length != 0:
            fail = s_count/length
            # fail = count[c]/length
        else:
            fail = 0
        fail_rate.append((c, fail))
        # length -= count[c]
        length -= s_count
    print(fail_rate)
    fail_rate = sorted(fail_rate, key=lambda x: x[1], reverse=True)
    answer = [i[0] for i in fail_rate]

    return answer

N = 5
stages = [2,1,2,6,2,4,3,3]
# stages = [4,4,4,4,4]
print(solution(N, stages))