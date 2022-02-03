import re

#match
p = re.compile('[a-z]+') #패턴 객체 생성, a~z까지 1번 이상 반복되는 문자열 찾는 식
m = p.match('3 python')
n = p.match('3 python')
# print(m) #매치가 되면 매치 객체를 돌려줌
# print(n) #매치가 안 되면 none 출력

#search
p = re.compile('[a-z]+')
m = p.search('3 python')
# print(m) #match와 달리 3이 있어도 일치하는 부분이 있으면 객체 돌려줌

#findall
p = re.compile('[a-z]+')
m = p.findall('life is too short')
# print(m) #일치하는 것을 찾아서 list로 출력

n = p.finditer('life is too short')
for r in n:
    # print(r) #for문을 통해서 매치되는 매치 객체 형태로 출력


#match method
    p = re.compile('[a-z]+')
    m = p.match('python')
    # print(m.group()) #python -> 매치된 문자열 나옴
    # print(m.start()) #첫 시작 인덱스
    # print(m.end()) #끝 인덱스
    # print(m.span()) #시작과 끝을 튜플 형태로 리턴


#DOTALL, S(약어)
p = re.compile('a.b', re.DOTALL)
m = p.match('a\nb')
# print(m) # '.'문자가 줄바꿈 문자도 포함하도록 하는것

#IGNORECASE, I
p = re.compile('[a-z]',re.I) #a~z 소문자만 가능
# print(p.match('python'))
# print(p.match('Python'))
# print(p.match('PYTHON'))    #대소문자를 모두 무시하고 모두 매치

#MULTILINE, M
p = re.compile("^python\s\w+",re.M) #^:각 라인의 맨 처음을 뜻함
                                    #\s: 공백을 나타내는 문자
                                    #\w: (w:word) 단어가 반복되는

data = """python one 
life is too short
python two
you need python
python three
python script language"""
# print(p.findall(data))

#VERBOSE, X #긴 정규식 표현을 나눠서 쓸 수 있게
charref = re.compile(r'&[#](0[0-7]+|[0-9]+|x[0-9a-fA-F]+);')

charref = re.compile(r"""
&[#]
(
    0[0-7]+
    |+[0-9]+
    |x[0-9a-fA-F]+
)
;
""")

#백슬래시 \

p = re.compile('\\section') (X)
p = re.compile('\\\\setction') (ㅇ)
p = re.compile(r'\\section') (==) #r'~' raw string


