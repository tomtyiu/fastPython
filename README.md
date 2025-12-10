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
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][https://github.com/tomtyiu/fastPython/fork]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][https://github.com/tomtyiu/fastPython/issues]
[![project_license][license-shield]][https://github.com/tomtyiu/fastPython/blob/main/LICENSE]



<!-- PROJECT LOGO -->
<br />
<!--
<div align="center">
  <a href="https://github.com/tomtyiu/fastPython">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h3 align="center">FastPython</h3>

  <p align="center">
    High Performance Python
    <br />
    <a href="https://github.com/tomtyiu/fastPython"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/tomtyiu/">View Demo</a>
    &middot;
    <a href="https://github.com/tomtyiu/fastPython/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/tomtyiu/fastPython/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
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
The FastPython is project is intended to increasae performance of python.  The focus is to provide Python code to run the python faster.
Most code will use local variables and avoid long dotted methods. Code will optimize for speed for different functions.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* Python

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

```py
python --version
```


### Prerequisites

Recommend Python 3.14.2
* python
  ```py
  python -m pip install --upgrade pip
  ```

### Installation

1. Download Python - [Download Python](https://www.python.org/downloads/)
2. Select python-3.14.2
3. Run python-3.14.2-amd64.exe 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example: 

```py
# Normal code
import math
val = math.sqrt(60)

#Total time taken: 7.499999810534064e-06

# fast code
from math import sqrt
val = sqrt(60)

#Total time taken: 6.599999323952943e-06
```
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


See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

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


