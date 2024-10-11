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
