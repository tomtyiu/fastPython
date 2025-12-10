#Accessing local variables is faster than globals because Python resolves them more efficiently.

def sum_fast(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
