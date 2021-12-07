N = 5
stages = [2,1,2,6,2,4,3,3]
# stages = [4,4,4,4,4]


def solution(N, stages):
    player_cur_stage = [0]*(N+1)
    total_clear = [0]*(N+1)
    for i in stages:
        player_cur_stage[i-1] +=1
        for j in range(i):
            total_clear[j] +=1

    failure_rate = []
    for i, (c,t) in enumerate(zip(player_cur_stage, total_clear)):
        if t != 0 and i!=N:
            failure_rate.append((i+1,c/t))
        elif t==0 and i!=N:
            failure_rate.append((i+1, 0))
        else: pass

    answer = sorted(failure_rate, key=lambda x: (-x[1],x[0]))
    answer = [i[0] for i in answer]
    return answer

print(solution(N, stages))









