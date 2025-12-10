def compute_squares(nums):
    append = list.append  # local reference to speed up method lookup
    result = []
    for n in nums:
        append(result, n * n)
    return result
