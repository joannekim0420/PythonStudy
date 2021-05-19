#using binary_search
def binary_search(array, start, end, target):
    while start <= end:
        mid = (start+end)//2

        if array[mid] == target:
            return mid
        elif array[mid] > target:
            end = mid-1
        else:
            start = mid+1
    return None

n = int(input())
inventory = list(map(int, input().split()))
inventory.sort()
m = int(input())
want = list(map(int, input().split()))

for t in want:
    result = binary_search(inventory, 0, n-1, t)
    if result == None:
        print('no', end=' ')
    else:
        print('yes', end=' ')


#without using binary_search
n = int(input())
# array = list(map(int, input().split()))
array = set(map(int, input().split()))
m = int(input())
x = list(map(int, input().split()))

for i in x:
    if i in array:
        print('yes', end= ' ')
    else:
        print('no', end=' ')