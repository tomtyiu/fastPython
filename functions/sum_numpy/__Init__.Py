from numpy import array, sum as np_sum

__all__ = ["sum_numpy"]

def sum_numpy(nums):
    arr = array(nums)
    return np_sum(arr)
