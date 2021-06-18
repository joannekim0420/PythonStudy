from itertools import combinations

a = [1,2,3,4]
b = [1,2,7,6,4]

def is_prime_number(sum):
    if sum ==0 or sum == 1:
        return False
    else:
        for n in range(2, (sum//2)+1):
            if sum%n == 0:
                return False
    return True

def solution(nums):
    answer = -1

    comb = list(combinations(nums, 3))
    count = 0
    for c in comb:
        if is_prime_number(sum(c)):
            count += 1

    answer = count

    return answer

print("sola:",solution(a))
print("solb:",solution(b))
