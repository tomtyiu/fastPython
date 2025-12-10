import time
from functools import lru_cache

#Leverage functools.lru_cache for Repeated Calls
@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
