import math


def solution(numbers, hand):
    answer = ''
    dist_dict = {1: [0, 0], 2: [1, 0], 3: [2, 0],
                 4: [0, 1], 5: [1, 1], 6: [2, 1],
                 7: [0, 2], 8: [1, 2], 9: [2, 2],
                 '*': [0, 3], 0: [1, 3], '#': [2, 3]}

    left_num = [1, 4, 7, "*"]
    right_num = [3, 6, 9, "#"]

    right_now = "#"
    left_now = "*"

    for num in numbers:
        if num in left_num:
            answer += "L"
            left_now = num

        elif num in right_num:
            answer += "R"
            right_now = num

        else:
            right_pos = dist_dict[right_now]
            left_pos = dist_dict[left_now]
            push_pos = dist_dict[num]

            # 오른쪽 거리가 왼쪽보다 멀 때
            if math.ceil(math.dist(right_pos, push_pos)) > math.ceil(math.dist(left_pos, push_pos)):
                answer += "L"
                left_now = num
            elif math.ceil(math.dist(right_pos, push_pos)) == math.ceil(math.dist(left_pos, push_pos)):
                answer += hand[0].upper()
                if hand[0].upper() == "R":
                    right_now = num
                else:
                    left_now = num
            else:
                answer += "R"
                right_now = num

    return answer