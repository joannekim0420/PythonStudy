def binary_search_dfs(array,target, start,end):
    middle_idx = (start+end)//2
    if start>end:
        return None
    if target == array[middle_idx]:
        return middle_idx
    elif target < array[middle_idx]:
        return binary_search_dfs(array, target, start, middle_idx-1)
    elif target > array[middle_idx]:
        return binary_search_dfs(array, target, middle_idx+1, end)

n, target = 10, 7
array = [1,3,5,7,9,11,13,15,17,19]

result = binary_search_dfs(array, target , 0, n-1)
print(result + 1)