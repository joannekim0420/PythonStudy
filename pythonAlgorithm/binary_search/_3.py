
def binary_search(array, start, end, target):
    if start>end:
        return None
    mid= (start+end)//2
    sum=0
    for index in array:
        if index-array[mid] <= 0:
            sum +=0
        else:
            sum += index-array[mid]

    if sum == target:
        print(array[mid])
        return mid
    elif sum > target:
        binary_search(array, mid+1, end, target)
    else:
        binary_search(array, start, mid-1, target)

n, m = map(int, input().split())
array = list(map(int, input().split()))
array.sort()

mid=binary_search(array, 0, n, m)
print(mid)
