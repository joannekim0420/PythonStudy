def check(data, array, i):
    if data == 'zero':
        array.append(0)
        return True
    elif data == 'one':
        array.append(1)
        return True
    elif data == 'two':
        array.append(2)
        return True
    elif data == 'three':
        array.append(3)
        return True
    elif data == 'four':
        array.append(4)
        return True
    elif data == 'five':
        array.append(5)
        return True
    elif data == 'six':
        array.append(6)
        return True
    elif data == 'seven':
        array.append(7)
        return True
    elif data == 'eight':
        array.append(8)
        return True
    elif data == 'nine':
        array.append(9)
        return True
    else:
        pass

def solution(s):
    answer = 0
    array = []
    for i in range(len(s)):
        if (ord(s[i]) >= 48 and ord(s[i]) <= 57):
            array.append(int(s[i]))
        else:
            for step in range(1, 6):
                prev = s[i:i+step]
                value = check(prev, array, i)
                if value == True:
                    break
    print(array)
    answer = int(''.join(map(str, array)))
    return answer
b = "23four5six7nine1one"

# b= "4seveneight1two"
print(solution(b))