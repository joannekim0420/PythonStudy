
#my first code
change = 1160

a=int(change/500)
b=int(change%500)
c=int(b/100)
d=int(b%100)
e=int(d/50)
f=int(d%50)
g=int(f/10)

sum = a+c+e+g
print('500원: {}, 100원: {}, 50원: {}, 10원: {}, sum: {}'.format(a,c,e,g,sum))


#solution
n=1260
count =0
coin_types = [500,100,50,10]
for coin in coin_types:
    count += n //coin
    n %=coin
print(count)
