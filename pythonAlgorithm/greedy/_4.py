#my code
def minus_one(n):
    return n-1

def divide_k(n,k):
    return n/k

n, k = map(int, input().split())
sum = 0
while(True):
    if(n%k==0):
        n= divide_k(n,k)
        sum+=1
    else:
        n= minus_one(n)
        sum+=1
    if(n==1):
        break
print(sum)
