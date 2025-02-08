# 1-task
a=25            #integer
b="Abdulaziz"   #string
c=25.0555       #float
d=True          #boolean
print(a, type(a))
print(b, type(b))
print(c, type(c))
print(d, type(d))

# 2-task
a=int(input("A ni kiriting: "))
b=int(input("B ni kiriting: "))
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a//b)
print(a%b)
print(a**b)

# 3-task
first_name=input("Ismingizni kiriting: ")
last_name=input("familiyangizni kiriting: ")
age=input("Yoshingizni kiriting: ")
ball=input("Ballingizni kiriting: ")
print(f"Ismi: {first_name}\nFamiliyasi: {last_name}\nYoshi: {age}\nOlgan bahosi : {ball}")

# 4-task
words=input("Sozni kiriting: ")
words=(words+"\n")*100
print(words)

# 5-task
words=input("So'zni kiriting: ")
print(*words, sep="<>")

# 6-task
word1=input("1 sozni kiriting: ")
word2=input("2 sozni kiriting: ")
word3=input("3 sozni kiriting: ")
print(f"{word1}\n{word2}\n{word3}")

# 7-task
word=input("So'zni kiriting: ")
nums=int(input("Sonni kiriting: "))
print(word[nums-1])

# 8-task
word=input("So'zni kiriting: ")
n=int(input("N sonini kiriting: "))
m=int(input("M sonini kiriting: "))
print(word[n:m+1])

# 9-task
word=input("So'zni kiriting: ")
nums=int(input("N sonni kiriting: "))
print(word[nums]*nums)