# Optimized vs lists
set_data = set(list_data)
another_set = set(another_list)
result = set_data - another_set

```
Operation: Converts both lists to sets and computes a set difference.

Time complexity:

Creating a set from a list is O(n).

Set difference (set_data - another_set) is roughly O(len(set_data)), because sets use hash tables for membership checks.

Overall: O(m + n) — linear.

Why faster:

Sets have O(1) average-time membership checks due to hashing.

No repeated scanning of the entire collection for each item.
```
