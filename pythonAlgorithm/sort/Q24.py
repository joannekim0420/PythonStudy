import statistics

n = int(input())
house_loc = list(map(int,input().split()))
house_loc.sort()
#sol1
print(house_loc[(n-1)//2])

#sol2
# min = 10000000
# min_sum = []
# min_index = 0
#
# for house in house_loc:
#     sum = 0
#     for i in house_loc:
#         sum += abs(house-i)
#     min_sum.append(sum)
#
# print(min_sum)
# for i in range(len(house_loc)):
#     if min > min_sum[i]:
#         min = min_sum[i]
#         min_index = house_loc[i]
#
# print(min_index)

