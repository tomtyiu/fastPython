#Use numpy vs range for faster execution time https://www.kdnuggets.com/speeding-up-your-python-code-with-numpy
import numpy as np
import time

sample = 1000000

list_1 = range(sample)
list_2 = range(sample)
start_time = time.time()
result = [(x + y) for x, y in zip(list_1, list_2)]
print("Time taken using Python lists:", time.time() - start_time)

array_1 = np.arange(sample)
array_2 = np.arange(sample)
start_time = time.time()
result = array_1 + array_2
print("Time taken using NumPy arrays:", time.time() - start_time)
