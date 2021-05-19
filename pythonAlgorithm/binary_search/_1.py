def sequential_search(n, target, array):
    for i in range(n):
        if array[i] == target:
            return i+1

input_data = input().split()
n = int(input_data[0])
target = input_data[1]

array = input().split()

print(sequential_search(n, target, array))

#binary search by recursive
def binary_search(array, target, start, end):
    if start>end:
        return None
    mid = (start+end)//2

    if array[mid] == target:
        return mid
    elif array[mid] > target:
        return binary_search(array, target, start, mid-1)
    else:     #array[mid] < target
        return binary_search(array, target, mid+1, end)

n, target = list(map(int, input().split()))
array = list(map(int, input().split()))

result = binary_search(array, target, 0, n-1)
if result == None:
    print("no value found")
else:
    print("result index:",result+1)


#binary search by while
def binary_search_w(array, target, start, end):
    while start<=end:
        mid = (start+end)//2

        if array[mid] == target:
            return mid
        elif array[mid] > target:
            end = mid-1
        else:
            start = mid+1
    return None

n , target = list(map(int, input().split()))
array = list(map(int, input().split()))

result = binary_search_w(array, target, 0, n-1)
if result == None:
    print("value not Found")
else:
    print("result index:", result+1)