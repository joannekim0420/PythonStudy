dict = {"abc":"xxx","abcd":"xyxy","cde":"yyy","cdefgh":"zzz"}

src = "qqq abcde rrr abcdefgh"

max_len = len(max(dict.items(), key=lambda x:len(x[0]))[0])
result = ""
i =0
now = 0

while i < len(src):
    length = min(len(src), i + max_len)
    tmp = ""
    # for j in range(min(len(src), i+max_len), i, -1):
    for j in range(i,length, 1):
        # print(src[i:j])
        if src[i:j] in dict:
            tmp = max(tmp, dict[src[i:j]])
            # i = j
        if length-j and now == i:
        #
            result += tmp
            i=j
        #
        #     now +=1
        #     break
        print(tmp)

            # result += dict[src[i:j]]
            # i = j
            # break

    else:
        result += src[i]
        i+=1
        now += 1

print(result)