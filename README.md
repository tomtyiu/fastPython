<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][https://github.com/tomtyiu/fastPython/fork]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][https://github.com/tomtyiu/fastPython/issues]
[![project_license][license-shield]][https://github.com/tomtyiu/fastPython/blob/main/LICENSE]
-->


<!-- PROJECT LOGO -->
<br />
<!--
<div align="center">
  <a href="https://github.com/tomtyiu/fastPython">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h3 align="center">FasterPython</h3>

  <p align="center">
    High Performance Python
    <br />
    <a href="https://github.com/tomtyiu/fasterPython"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/tomtyiu/">View Demo</a>
    &middot;
    <a href="https://github.com/tomtyiu/fasterPython/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/tomtyiu/fasterPython/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This is the official GitHub repository for FastPython.
FasterPython is an open-source project dedicated to improving Python’s performance through practical, drop-in code patterns and optimization techniques. The goal is simple: help Python run faster without sacrificing readability.

The project emphasizes the use of local variables, reduced attribute lookups, and other micro-optimizations that meaningfully improve execution speed. Each example focuses on accelerating specific functions or workloads.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



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
pip install fasterPython

#Example
pip install -i https://test.pypi.org/simple/ fasterpython==0.0.8
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
from fasterPython.fasterPython import fast_sqrt, fast_log

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
from fasterPython.fasterPython import FAST_PI

area = FAST_PI * r * r
```

---

### Cached Fibonacci

A cached recursive Fibonacci implementation using `functools.lru_cache`.
The lru_cache(maxsize=None) decorator in Python creates an unlimited cache that stores the results of all function calls without any eviction policy. This effectively disables the "Least Recently Used" (LRU) aspect of the cache, turning it into a simple memoization tool. 

```python
fib(n)
```

Example:

```python
from fasterPython.fasterPython import fib

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
from fasterPython.fasterPython import sum_fast

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
from fasterPython.fasterPython import compute_squares

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
from fasterPython.fasterPython import sum_numpy

sum_numpy(range(1_000_000))
```

## Bubble Sort (fast)
```python
#Fast Bubble Sort 
mylist = [64, 34, 25, 12, 22, 11, 90, 5]

n = len(mylist)
# cache the list length in a local variable
for i in range(n - 1):
    # reduce the range as the largest elements bubble to the end
    for j in range(n - i - 1):
        a, b = mylist[j], mylist[j + 1]  # local variable assignment
        if a > b:
            mylist[j], mylist[j + 1] = b, a

print(mylist)
```
| Operation | Standard sort | Fast Sort | Improvement |
|---------|------------------|----------|-------------|
| `BubbleSort` | ~20 ns | ~11 ns | ~45% |


## Use multiprocessing, multiprocessing is a package that supports spawning processes using an API similar to the threading module.
```py
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, range(10)))
```
| Method                   | Execution Time (seconds)       |
|--------------------------|-------------------------------|
| Normal                   | 0.022988299999269657          |
| Multiprocessing          | 0.0000045000015234109014      |
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

## Benchmarks

Benchmarks were run on Python 3.12, Windows 11, x86-64 CPU, using `timeit` with 1,000,000 iterations unless otherwise noted. Results are representative, not absolute.

### Scalar Operations

| Operation | Standard `math` | FastMath | Improvement |
|---------|------------------|----------|-------------|
| `sqrt(x)` | ~85 ns | ~65 ns | ~23% |
| `sin(x)`  | ~95 ns | ~72 ns | ~24% |

The improvement comes primarily from reduced attribute lookup (`math.sqrt` → local binding).

### Summation (10,000 integers)

| Method | Time |
|------|------|
| `sum()` | ~78 µs |
| `sum_fast()` | ~62 µs |
| `sum_numpy()` | ~180 µs |

For small to medium iterables, pure Python outperforms NumPy due to array construction overhead.

### Fibonacci (n = 40)

| Implementation | Time |
|---------------|------|
| Naive recursion | >10 seconds |
| Cached `fib()` | ~0.4 ms |

Caching converts an exponential problem into a linear one.

---

## License

MIT License. Use it freely, modify it boldly, and benchmark everything.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Tips

1. **Profile to find bottlenecks**
   - Use `cProfile` for whole‑program, function‑level stats:
     `python -m cProfile -o stats.out your_script.py`, then inspect with `pstats` or tools like Snakeviz.
   - Use `line_profiler` or Scalene for line‑level hotspots.
   - Use `timeit` for micro‑benchmarks when comparing small code variants.
   Profiling ensures you optimize the parts that actually matter. 【ymichael.com†】【jun-yan.github.io†】【devopedia.org†】【dev.to†】

2. **Exploit faster built‑ins and data structures**
   - Prefer built‑ins (`sum`, `min`, `max`, `sorted`, `any`, `all`) and `itertools` over manual loops; they run in optimized C.
   - Choose the right container: use `dict`/`set` for O(1) average lookups instead of repeatedly scanning lists. 【geeksforgeeks.org†】

3. **Make loops and list operations cheaper**
   - Use list comprehensions and generator expressions instead of `for` + `append`.
   - Hoist loop‑invariant work outside the loop and avoid repeated global/attribute lookups; bind to locals when a function is used in a tight loop
.
   - Iterate directly over items (`for x in a`) and use `zip` for multiple sequences.
   - Avoid repeated string concatenation in loops; accumulate pieces in a list and `''.join(...)` once. 【geeksforgeeks.org†】【en.wikipedia.org†】


4. **Improve algorithms, not just syntax**
   - Replace quadratic algorithms with linear or `O(n log n)` ones via better data structures or sorting.
   - Cache expensive pure functions with `functools.lru_cache` to avoid recomputation.

5. **Use vectorization and compiled paths for numeric code**
   - Replace Python loops over numbers with NumPy vectorized operations; these run in fast C code and can be 10–100× faster.
   - For complex numeric loops that are hard to vectorize, use:
     - **Numba**: add `@njit` to numeric, NumPy-heavy functions to JIT‑compile them.
     - **Cython**: add static types and compile critical modules to C for maximum control.
   Both can deliver order‑of‑magnitude speedups on tight numeric loops. 【medium.com†】【c-sharpcorner.com†】【cbtw.tech†】【geeksforgeeks.org†】

6. **Environment choices**
   - Use a recent CPython (3.11+) for built‑in speedups.
   - For long‑running, pure‑Python workloads, consider PyPy or offloading hot sections to C/C++/Rust through Cython or dedicated libraries.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] basic python functions and AI
- [ ] advance python functions and loops


See the [open issues](https://github.com/tomtyiu/fasterPython/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:
TBD



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Thomas Yiu - [@twitter_handle](https://twitter.com/twitter_handle) - tom.tyiu@gmail.com

Project Link: [https://github.com/tomtyiu/fastPython](https://github.com/tomtyiu/fastPython)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [OpenAI](https://openai.com/)
* [Anthropic](https://www.anthropic.com/)
* [Python.org ](https://www.python.org/)
* [Github](https://github.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 


