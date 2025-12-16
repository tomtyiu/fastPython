# FastMath Utilities

FastMath Utilities is a lightweight Python module that provides fast-path wrappers around commonly used mathematical operations, constants, and performance-oriented patterns. The goal is not to replace Python’s standard library, but to expose performance-friendly access patterns that reduce overhead in tight loops, hot paths, and numerical workloads.

This module focuses on:
- Minimizing attribute lookups
- Avoiding repeated imports
- Using local bindings for speed
- Providing optional NumPy acceleration
- Keeping behavior predictable and explicit

It is designed for Python 3.10+ and works especially well in performance-sensitive code.

---

## Design Philosophy

Python performance is often limited not by computation itself, but by overhead:
- Repeated dotted lookups such as `math.sqrt`
- Function indirection
- Recomputing the same values
- Namespace pollution

FastMath Utilities removes these costs by:
- Binding frequently used math functions once
- Exposing constants directly instead of via functions
- Using caching where recursion is unavoidable
- Keeping implementations readable and debuggable

There is no hidden magic here. The speed comes from removing friction.

---

## Installation

This module can be vendored directly into a project or packaged as part of a larger performance-focused library.

```bash
pip install fastmath-utils
```

If copied directly into your codebase, no installation step is required.

---

## API Reference

### Scalar Math Functions

Thin wrappers around `math` functions with minimal overhead.

```python
fast_sqrt(x)
fast_ceil(x)
fast_floor(x)
fast_factorial(x)
fast_log(x, base=None)
fast_exp(x)
fast_sin(x)
```

Example:

```python
from fastmath import fast_sqrt, fast_log

fast_sqrt(25)        # 5.0
fast_log(8, 2)       # 3.0
fast_log(10)         # natural logarithm
```

---

### Mathematical Constants

Constants are exposed directly to avoid function call overhead.

```python
FAST_PI
FAST_E
FAST_TAU
FAST_INF
FAST_NAN
```

Example:

```python
from fastmath import FAST_PI

area = FAST_PI * r * r
```

---

### Cached Fibonacci

A cached recursive Fibonacci implementation using `functools.lru_cache`.

```python
fib(n)
```

Example:

```python
from fastmath import fib

fib(40)  # computed once and reused
```

This avoids the exponential blow-up of naive recursion and is suitable for repeated calls.

---

### Fast Pure-Python Sum

A hand-rolled summation loop that avoids function call overhead and shadowing built-ins.

```python
sum_fast(iterable)
```

Example:

```python
from fastmath import sum_fast

sum_fast([1, 2, 3, 4])
```

This is ideal for small to medium iterables where NumPy would introduce unnecessary overhead.

---

### Optimized List Computation

Efficient square computation using local method binding to reduce attribute lookup cost.

```python
compute_squares(nums)
```

Example:

```python
from fastmath import compute_squares

compute_squares([1, 2, 3, 4])
```

---

### NumPy-Accelerated Sum (Optional)

If NumPy is installed, a fast vectorized summation function is available.

```python
sum_numpy(nums)
```

Example:

```python
from fastmath import sum_numpy

sum_numpy(range(1_000_000))
```

If NumPy is not installed, calling this function raises a clear runtime error instead of failing silently.

---

## When to Use This Module

FastMath Utilities is useful when:
- Optimizing inner loops
- Writing performance-sensitive Python code
- Building numerical or agentic systems
- Reducing micro-overhead without sacrificing clarity

It is not intended to replace NumPy, SciPy, or compiled extensions for large-scale numerical workloads.

---

## Performance Notes

- These wrappers do not outperform `math`; they reduce lookup and call overhead.
- Constants are faster than functions returning constants.
- NumPy is faster for large arrays; pure Python is often faster for small iterables.
- Readability is preserved intentionally—fast code that cannot be understood is a liability.

---

## License

MIT License. Use it freely, modify it boldly, and benchmark everything.

