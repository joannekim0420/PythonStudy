
def check_correct(s):
    if "()" in s:
        tmp = s.split("()")
        string = ""
        for i in range(len(tmp)):
            string += tmp[i]
        check_correct(string)
    else:
        if s == "":
            print("True")
            return True
        else:
            return False



# check_correct("(()())()")

a= check_correct("()))((()")
if a == True:
    print("T")
elif a == False:
    print("F")
else:
    print("Error")