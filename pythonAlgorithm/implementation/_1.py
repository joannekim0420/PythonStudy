#my answer code
n = int(input())
path = list(map(str, input().split()))

xpoint, ypoint = 1,1
for p in path:
    if(p=='R'):
        ypoint+=1
        if(ypoint>n):
            ypoint-=1
    elif(p=='L'):
        ypoint-=1
        if (ypoint < 1):
            ypoint += 1
    elif(p=='U'):
        xpoint-=1
        if (xpoint < 1):
            xpoint += 1
    else:
        xpoint+=1
        if (xpoint > n):
            xpoint -= 1

print("Xpoint: {} Ypoint: {}".format(xpoint, ypoint))

# # answer code
# x,y,=1,1
# plans = input().split()
#
# dx = [0,0,-1,1]
# dy=[-1,1,0,0]
# move_types = ['L','R','U','D']
#
# for plan in plans:
#     for i in range(len(move_types)):
#         if plan == move_types[i]:
#             nx=x+dx[i]
#             ny=y+dy[i]
#     if nx <1 or ny <1 or nx> n or nx> n:
#         continue
#     x,y = nx, ny
#
# print(x,y)