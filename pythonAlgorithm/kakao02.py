places = [["POOOP", "OXXOX", "OPXPX", "OOXOX", "POXXP"],
          ["POOPX", "OXPXP", "PXXXO", "OXXXO", "OOOPP"],
          ["PXOPX", "OXOXP", "OXPXX", "OXXXP", "POOXX"],
          ["OOOXX", "XOOOX", "OOOXX", "OXOOX", "OOOOO"],
          ["PXPXP", "XPXPX", "PXPXP", "XPXPX", "PXPXP"]]

# print(len(places))
def mdistance(a,b):
    x1,y1 = a
    x2, y2 = b
    return (abs(x1-x2) + abs(y1-y2))

def to_coordinations(p, graph):
    # print("p:",p)
    array = []
    for i in range(5):
        for j in range(5):
            if p[i][j] == 'P':
                graph[i][j] = (j,i)
                array.append((j,i))
            elif p[i][j] =='X':
                graph[i][j] = 1
            else:
                graph[i][j] = 0
    # print("graph",graph)
    print("Array:",array)
    if bool(array) == False:
        return 1
    for i in range(len(array)-1):
        for j in range(i+1,len(array)):
            print(array[i],array[j])
            dist = mdistance(array[i],array[j])
            print(dist)
            if dist<=2:
                return dist

def solution(places):
    answer = []
    graph = [[0 for x in range(5)] for y in range(5)]

    for p in places:
        value = to_coordinations(p, graph)
        if value<=2:
            answer.append(0)
        else:
            answer.append(1)

    return answer

print(solution(places))