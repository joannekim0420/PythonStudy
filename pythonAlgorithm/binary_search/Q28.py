n = int(input())

data = list(map(int, input().split()))

def binary_search_dfs(array, start, end):
    ##mid = target
    mid = (start+end)//2
    if start>end:
        return -1
    if array[mid] == mid:
        return mid
    elif array[mid] < mid:
        return binary_search_dfs(array, mid+1, end)
    elif array[mid] > mid:
        return binary_search_dfs(array, start, mid-1)
print(binary_search_dfs(data, 0, n-1))