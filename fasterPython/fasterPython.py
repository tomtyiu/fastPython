# math fast-path bindings
from math import (
    sqrt,
    ceil,
    floor,
    factorial,
    log,
    exp,
    sin,
    pi,
    e,
    tau,
    inf,
    nan,
)

# ------------------
# Scalar math funcs
# ------------------

def fast_sqrt(x): return sqrt(x)
def fast_ceil(x): return ceil(x)
def fast_floor(x): return floor(x)
def fast_factorial(x): return factorial(x)
def fast_log(x, base=None): return log(x) if base is None else log(x, base)
def fast_exp(x): return exp(x)
def fast_sin(x): return sin(x)

# ------------------
# Math constants
# ------------------

FAST_PI = pi
FAST_E = e
FAST_TAU = tau
FAST_INF = inf
FAST_NAN = nan

# ------------------
# Cached recursion
# ------------------

from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

# ------------------
# Fast pure-Python sum
# ------------------

def sum_fast(iterable):
    total = 0
    for v in iterable:
        total += v
    return total

# ------------------
# Avoid repeated lookups
# ------------------

def compute_squares(nums):
    result = []
    append = result.append  # local binding
    for n in nums:
        append(n * n)
    return result

# ------------------
# NumPy fast path (optional)
# ------------------

try:
    import numpy as _np

    def sum_numpy(nums):
        return _np.sum(_np.asarray(nums))

except ImportError:
    def sum_numpy(nums):
        raise RuntimeError("NumPy is not installed")
