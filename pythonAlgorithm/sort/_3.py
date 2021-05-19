student_Score = []

n = int(input())

for i in range(n):
    student_Score.append(input().split())

student_Score.sort()

#1
for i in range(len(student_Score)):
    print(student_Score[i][0], end = ' ')

#2
for student in student_Score:
    print(student[0], end= ' ')