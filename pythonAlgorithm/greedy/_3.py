m,n= map(int,input().split())

# #my answer
# min = []
# for i in range(n):
#     data = list(map(int, input().split()))
#     data.sort()
#     min.append(data[0])
# min.sort()
# print(min[n-1])
#
#
# result = 0
# for i in range(n):
#     data = list(map(int,input().split()))
#     # answer1
#     min_value = min(data)
#     result = max(result,min_value)
#
#     # answer2
#     min_Value = 100001
#     for a in data:
#         min_Value = min(min_Value, a)
#     result = max(result, min_Value)
#
# print(result)
