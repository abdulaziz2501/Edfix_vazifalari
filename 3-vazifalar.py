# # 1-task
# lst = [1, 2, 33, 5, 6, 7, 7]
# n=int(input("Sonni kiriting: "))
# for i in range(len(lst)):
#     for j in range(i+1, len(lst)):
#         if lst[i]+lst[j]==n:
#             print(i, j)
#
# # 2-task
# txt=input("matnni kiriting: ")
# lst=txt.split(" ")
# lst.sort()
# print(' '.join(lst))
#
# # 3-task
# n=int(input("Sonni kiriting: "))
# summa=0
# for i in range(1, n+1):
#    if i==n:
#        print(f"{i}={summa+i}", end="")
#        break
#    print(f"{i} + ", end="")
#    summa += i
#
# # 4-task
# txt=input("so'zni kiriting: ")
# lst=txt.split(" ")
# for i in range(len(lst)):
#     if len(lst[i])%2:
#         lst[i]=lst[i][::-1]
# print(' '.join(lst))
#
# # 5-task
# num=int(input("Sonni kiriting: "))
# s=bin(num)[2:]
# count=0
# for i in range(len(s)):
#     if s[i]=='0':
#         count+=1
# print(count)
#
# # 6-task
# pass
#
# # 7-task
# txt=input("matnni kiriting: ")
# character=input("harfni kiriting: ")
# if character.islower() and len(character)==1:
#     for i in range(len(txt)):
#         if character in txt:
#             txt[i].upper()
# print(txt)
#hali to'liq emas 6-7 masala
#
# # 8-task
# n=int(input("Sonni kiriting: "))
# count=0
# for i in range(1, n+1):
#     if i == n:
#         print(f"{int('2'*i)}={count+int('2'*i)}", end="")
#         break
#     print(f"{int('2'*i)} + ", end="")
#     count += int('2'*i)
#
# # 9-task
pass

# 10-task
n=int(input("sonni kiriting: "))
a=len(str(n))
b=str(n)
for i in range( a):
    d=str(int(b[i])*10**(a-i-1))
    if i == n:
        print(f"{d}={d}", end="")
        break
    print(f"{d} + ", end="")
