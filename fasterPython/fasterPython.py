from math import sqrt
def fast_sqrt(x):
    return sqrt(x)

from math import ceil
def fast_ceil(x):
    return ceil(x)

from math import floor
def fast_floor(x):
    return floor(x)

from math import pi
def fast_pi():
    return pi

from math import factorial
def fast_factorial():
    return factorial(x)

from math import log
def fas_log(x,y):
    return log(x, y)

from math import exp
def fast_exp(x):
    return exp(x)

from math import sin
def fast_sin(x):
    return sin(x)

from math import e
def fast_e():
    return e

from math import tau
def fast_tau():
    return tau

from math import inf
def fast_inf():
    return inf


from math import nan
def fast_nan():
    return nan

import time
from functools import lru_cache

#Leverage functools.lru_cache for Repeated Calls
@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

def sum_fast(x):
    total = 0
    for num in x:
        total += num
    return total

#Avoid Repeated Computations
#Store results instead of recalculating.

def compute_squares(nums):
    append = list.append  # local reference to speed up method lookup
    result = []
    for n in nums:
        append(result, n * n)
    return result

# fast numpy
from numpy import array, sum

def sum_numpy(nums):
    arr = array(nums)
    return sum(arr)

import os
from openai import OpenAI


def openai_api(prompt):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    create = client.responses.create
    responses = create(
        model="gpt-5.2",
        instructions="You are a coding assistant that talks like a pirate.",
        input=prompt,
    )

#print out response
# responses=response.output_text
# print(responses)
