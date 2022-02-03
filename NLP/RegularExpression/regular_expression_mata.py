#메타 문자
import re

# | or의 의미
p = re.compile('Crow|Servo')
m = p.match('CrowHello')
m = p.match('ServoHello')
print(m) #Crow나 Servo 일치하는 문자 있으면 매치

