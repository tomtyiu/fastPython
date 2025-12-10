# Built-in Math Functions
### For python to run faster than normal, please use build-in math functions

Python includes several built-in functions for basic mathematical operations:

min() and max(): These functions return the smallest and largest values in an iterable, respectively.

```py
x = min(5, 10, 25)
y = max(5, 10, 25)
print(x) # Output: 5
print(y) # Output: 25
```


abs(): This function returns the absolute (positive) value of a specified number.
```py
x = abs(-7.25)
print(x) # Output: 7.25
```

pow(x, y): This function returns the value of x raised to the power of y.
```py
x = pow(4, 3)
print(x) # Output: 64
```

# The Math Module

The math module extends the list of mathematical functions available in Python. To use it, you must import the module:

```
import math
```

Common Functions in the Math Module

math.sqrt(x): Returns the square root of a number.
```
x = math.sqrt(64)
print(x) # Output: 8.0
```


math.ceil(x): Rounds a number upwards to its nearest integer.
```py
x = math.ceil(1.4)
print(x) # Output: 2
```

math.floor(x): Rounds a number downwards to its nearest integer.
```py
x = math.floor(1.4)
print(x) # Output: 1
```

math.pi: Returns the value of PI (3.141592...).
```py
x = math.pi
print(x) # Output: 3.141592653589793
```

math.factorial(n): Returns the factorial of a number.
```
x = math.factorial(5)
print(x) # Output: 120
```

math.log(x, base): Returns the logarithm of x to the given base.
```
x = math.log(100, 10)
print(x) # Output: 2.0
```

math.exp(x): Returns e raised to the power of x.
```
x = math.exp(2)
print(x) # Output: 7.3890560989306495
```

math.sin(x), math.cos(x), math.tan(x): Return the sine, cosine, and tangent of x radians, respectively.
```
x = math.sin(math.pi / 2)
print(x) # Output: 1.0
```

# Constants in the Math Module
math.e: The mathematical constant e (2.718281...).
```
x = math.e
print(x) # Output: 2.718281828459045
```

math.tau: The mathematical constant τ (6.283185...), which is equal to 2π.
```
x = math.tau
print(x) # Output: 6.283185307179586
```

math.inf: A floating-point positive infinity.
```
x = math.inf
print(x) # Output: inf
```

math.nan: A floating-point "not a number" (NaN) value.

x = math.nan
print(x) # Output: nan
