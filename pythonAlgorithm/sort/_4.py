n, k = map(int, input().split())

listA = list(map(int, input().split()))
listB = list(map(int, input().split()))

listA.sort()
listB.sort(reverse=True)

for i in range(k):
    if listA[i] < listB[i]:
        listA[i], listB[i] = listB[i], listA[i]

max_sum = sum(listA)

print(max_sum)
