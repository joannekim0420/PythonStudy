#Fibonacci Function

def simple_fibo(x):
    if x==1 or x==2:
        return 1
    return simple_fibo(x-1)+simple_fibo(x-2)

# memoization fibo function
d = [0]*100
def memoization_fibo(x):
    # print('f('+str(x)+')',end=' ')
    if x==1 or x==2:
        return 1
    if d[x]!=0:
        return d[x]
    d[x] = memoization_fibo(x-1)+memoization_fibo(x-2)
    return d[x]

# bottom-up fibo

d = [0]*100
d[1] = 1
d[2] = 1

n=99
for i in range(3, n+1):
    d[i] = d[i-1] + d[i-2]
print(d[n])

