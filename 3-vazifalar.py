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
# son = int(input("Son kiriting: "))
# count = 0
# current = son + 1
# lst = []
# while count < 5:
#     tub = True
#     for i in range(2, current):
#         if current % i == 0:
#             tub = False
#             break
#     if tub:
#         lst.append(current)
#         count += 1
#
#     current += 1
# print(*lst)

#
# # 7-task
# txt=input("matnni kiriting: ")
# character=input("harfni kiriting: ")
# natija = ''.join([char.upper() if char == character else char for char in txt])
# print(natija)
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
# for num in range(1000, 10000):
#     minglik = num // 1000
#     yuzlik = (num // 100) % 10
#     onlik = (num // 10) % 10
#     birlik = num % 10
#     if minglik != yuzlik and minglik != onlik and minglik != birlik and yuzlik != onlik and yuzlik != birlik and onlik != birlik:
#         print(num)
#
# # 10-task
# n=int(input("sonni kiriting: "))
# a=len(str(n))
# b=str(n)
# for i in range( a):
#     d=str(int(b[i])*10**(a-i-1))
#     if i+1==a:
#         print(f"{d}", end="")
#         break
#     print(f"{d} + ", end="")
#
# # 11-task
# num = int(input("Sonni kiriting: "))
# lst = []
# for i in range(2, num + 1):
#     while num % i == 0:
#         lst.append(i)
#         num //= i
# print(lst)
#
# # 12-task
# lst=[1, 2, 3]
# a = []
# n = len(lst)
# for i in range(n):
#     for j in range(n):
#         for k in range(n):
#             if i != j and i != k and j != k:
#                 results = [lst[i], lst[j], lst[k]]
#                 a.append(results)
# print(a)
#
# # 13-task
# lst = [1, 3, 4, 9, 3, 4, 0, -1, 7, 8]
# a = []
# b = []
# for i in range(len(lst)):
#     if not b or lst[i] > b[-1]:
#         b.append(lst[i])
#     else:
#         if b:
#             a.append(b)
#             b = []
#         b.append(lst[i])
# if b:
#     a.append(b)
# for i in a:
#     print(i)
# print("O'sish tartibida kelgan to'plamlar soni:", len(a))
