array = [7,5,9,0,3,1,6,2,4,8]

print(array)
for i in range(len(array)):
    for j in range(i+1,len(array)):
        if array[j] < array[i]:
            # tmp = array[j]
            # array[j] = array[i]
            # array[i] = tmp
            array[i], array[j] = array[j], array[i]
print(array)