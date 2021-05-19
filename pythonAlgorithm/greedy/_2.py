import time
n, m, k = map(int, input().split())

data = list(map(int, input().split()))

start = time.time()
data.sort()
f_biggest = data[n-1]
s_biggest = data[n-2]
result =0

# my answer
# t = 1
# while t<=m:
#     if(t%(k+1)==0):
#         result+=s_biggest
#     else:
#         result += f_biggest
#     t+=1

# answer1
# while True:
#     for i in range(k):
#         if m==0:
#             break
#         result += f_biggest
#     if m ==0:
#         break
#     result += s_biggest
#     m-=1

# answer2
# count = int(m/(k+1))*k
# count += m% (k+1)
#
# result += (count)*f_biggest
# result += (m-count) * s_biggest

end = time.time()
print('result: {}, duration: {}' .format(result, end-start))