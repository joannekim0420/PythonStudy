n = input()

array = [ord(c) for c in n]
alphabet = []
number = []

for i in array:
    if i >= 65 and i<=90:
        alphabet.append(chr(i))
    elif i>=48 and i<=57:
        number.append(int(chr(i)))
    else:
        break
alphabet.sort()

for i in alphabet:
    print(i,end='')
print(sum(number))
