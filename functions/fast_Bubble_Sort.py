#Fast Bubble Sort 
mylist = [64, 34, 25, 12, 22, 11, 90, 5]

n = len(mylist)
for i in range(n - 1):
    swapped = False  # track if any swaps occurred
    for j in range(n - i - 1):
        a, b = mylist[j], mylist[j + 1]
        if a > b:
            mylist[j], mylist[j + 1] = b, a
            swapped = True
    if not swapped:
        break  # list is sorted early

print(mylist)
