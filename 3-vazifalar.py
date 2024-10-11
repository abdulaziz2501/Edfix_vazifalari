lst = [1, 2, 33, 5, 6, 7, 7]
n=int(input("Sonni kiriting: "))
for i in range(len(lst)):
    for j in range(i+1, len(lst)):
        if lst[i]+lst[j]==n:
            print(i, j)
