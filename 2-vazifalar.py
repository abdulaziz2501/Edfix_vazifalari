# 1-task
from jedi.settings import fast_parser

num=int(input("Sonni kiriting: "))
if num%3==0 and num%5==0:
    print("FizzBuzz")
elif num%3==0:
    print("Fizz")
elif num%5==0:
    print("Buzz")

# 2-task
num=int(input("Sonni kiriting: "))
if num%7==0:
    print(True)

# 3-task
num=int(input("Sonni kiriting: "))
if num>0:
    a="musbat"
if num<0:
    a="manfiy"
if num%2==0:
    b="Juft"
if num%2:
    b="Toq"
else:
    print("nol")
print(a,b)

# 4-task
a=int(input("A ni kiriting: "))
b=int(input("B ni kiriting: "))
if a%b==0 or b%a==0:
    print(True)
else:
    print(False)

# 5-task
burchak=int(input("BUrchakni kiriting: "))
if burchak>0 and burchak<90:
    print("O'tkir buruchak")
elif burchak==90:
    print("To'g'ri burchak")
elif burchak>90 and burchak<180:
    print("O'tmas burchak")
elif burchak==180:
    print("Yoyiq burchak")

# 6-task
num=int(input("Sonni kiriting: "))
if num%2:
    print(num-2)
else:
    print(num+2)

# 7-task
num=int(input("Sonni kiriting: "))
if num>=0 and num<10:
    print("raqam")
else:
    print("raqam emas")

# 8-task
num=int(input("Sonni kiriting: "))
if num%4==0 and num%5:
    print(True)
else:
    print(False)

# 9-task
\\