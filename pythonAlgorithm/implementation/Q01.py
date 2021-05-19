n = input()

sumR, sumL = 0,0
length = int(len(n)/2)

for i in range(0,length):
    sumL += int(n[i])
    sumR += int(n[length+i])
if sumL == sumR:
    print("LUCKY")
else:
    print("READY")