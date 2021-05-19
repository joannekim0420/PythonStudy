input_data = input()
row = int(input_data[1])
col = int(ord(input_data[0])) - int(ord('a'))+1

dx = [-2,2, -2,2, -1,1, -1,1]
dy = [1, 1, -1, -1, 2, 2, -2,-2]

x = col
y = row
count=0
for a,b in zip(dx, dy):
    nx = x+a
    ny = y+b
    if(nx>0 and nx<=8 and ny>0 and ny<=8):
        count+=1
print(count)