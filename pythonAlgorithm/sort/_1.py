import time
array1 = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

#my sort
for i in range(len(array1)):
    for j in range(i+1, len(array1)):
        if array1[j]<array1[i]:
            array1[j], array1[i] = array1[i], array1[j]
# print("my_sort:",array1)

#selection sort
array2 = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
for i in range(len(array2)):
    min_index = i
    for j in range(i+1, len(array2)):
        if array2[min_index] > array2[j]:
            min_index = j
    array2[i], array2[min_index] = array2[min_index], array2[i]
# print("selection_sort:",array2)

#insertion sort
array3 = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
for i in range(1, len(array3)):
    for j in range(i, 0, -1):
        if array3[j] < array3[j-1]:
            array3[j], array3[j-1] = array3[j-1], array3[j]
        else:
            break
# print("insertion_sort:",array3)

#quick sort
array4 = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
def quick_sort(array, start, end):
    if start >= end:
        return
    pivot = start
    left = start+1
    right = end
    while left<=right:
        while left<= end and array[left] <= array[pivot]:
            left +=1
        while right > start and array[right] >= array[pivot]:
            right -= 1

        if left > right:
            array[right], array[pivot] = array[pivot] , array[right]
        else:
             array[left], array[right] = array[right], array[left]

    quick_sort(array, start, right-1)
    quick_sort(array, right+1, end)

quick_sort(array4, 0, len(array4)-1)
# print("quick_sort:", array4)

#simple quick sort
array5 = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
def simple_quick_sort(array):
    if len(array)<=1:
        return array

    pivot = array[0]
    tail = array[1:]

    left_side = [x for x in tail if x<= pivot]
    right_side = [x for x in tail if x>pivot]

    return simple_quick_sort(left_side)+[pivot]+simple_quick_sort(right_side)

# print("simple quick sort:",simple_quick_sort(array5))

array6 = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]

count = [0]*(max(array6)+1) #계수 정보를 저장하는거니까 가장 큰 계수 정보 필요

for i in range(len(array6)):
    count[array6[i]] += 1

for i in range(len(count)):
    for j in range(count[i]):
        print(i, end=' ')

