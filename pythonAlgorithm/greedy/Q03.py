n = input()
count0, count1 =0,0
for i in range(len(n)-1):
    if int(n[i]) == 0 and int(n[i+1]) == 1:
        count0 += 1
    elif int(n[i]) == 1 and int(n[i+1]) == 0:
        count1 += 1
    else:
        continue

if count0 > count1:
    print(count0)
else:
    print(count1)