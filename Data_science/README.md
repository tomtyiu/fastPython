# Data science

## NumPy
Example code
```
import math
import time

sample = 1_000_000
list_1 = range(sample)

start_time = time.time()
result = [math.sin(x) for x in list_1]
print("Time using Python lists:", time.time() - start_time)

import numpy as np
import time

array_1 = np.arange(sample)

start_time = time.time()
result = np.sin(array_1)
print("Time using NumPy arrays:", time.time() - start_time)

```

Please use NumPy version

```
np.sin(array_1) i
```
vectorized, done in compiled C code, operating on the entire array at once and also uses SIMD instructions, CPU cache-friendly memory access.
