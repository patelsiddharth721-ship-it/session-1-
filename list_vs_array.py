"""
list_vs_array.py
Compare Python lists and NumPy arrays: memory, speed, and vectorization.

Setup:  pip install numpy
Run:    python list_vs_array.py
"""

import sys
import time

import numpy as np


# ---------------------------------------------------------------
# Task 1: Create a list and a NumPy array with numbers 1 to 1000
# ---------------------------------------------------------------
py_list = list(range(1, 1001))
np_array = np.arange(1, 1001)


# ---------------------------------------------------------------
# Task 2: Memory usage
# ---------------------------------------------------------------
def show_memory_usage():
    list_container = sys.getsizeof(py_list)
    # getsizeof on a list counts only the list object (its pointers),
    # not the int objects it points to. Add those for a fairer total.
    list_total = list_container + sum(sys.getsizeof(x) for x in py_list)

    print("=== Memory usage (1000 integers) ===")
    print(f"Python list (container only): {list_container} bytes")
    print(f"Python list (incl. int objects): {list_total} bytes")
    print(f"NumPy array (nbytes):          {np_array.nbytes} bytes")
    print(f"NumPy dtype: {np_array.dtype}")
    print()


# ---------------------------------------------------------------
# Task 3: Speed comparison - add 5 to every element (10,000 items)
# ---------------------------------------------------------------
def compare_addition_speed():
    n = 10_000
    big_list = list(range(1, n + 1))
    big_array = np.arange(1, n + 1)

    repeats = 100  # repeat to get a stable average (single runs are noisy)

    # Python list: loop through every element
    start = time.perf_counter()
    for _ in range(repeats):
        list_result = [x + 5 for x in big_list]
    list_time = (time.perf_counter() - start) / repeats

    # NumPy array: one vectorized operation
    start = time.perf_counter()
    for _ in range(repeats):
        array_result = big_array + 5
    array_time = (time.perf_counter() - start) / repeats

    print("=== Addition speed (add 5 to 10,000 integers) ===")
    print(f"Python list: {list_time:.8f} seconds (avg of 100 runs)")
    print(f"NumPy array: {array_time:.8f} seconds (avg of 100 runs)")
    if array_time > 0:
        print(f"NumPy was about {list_time / array_time:.1f}x faster")
    print()


# ---------------------------------------------------------------
# Task 4: Vectorization vs for-loop (Zomato-style ratings x 2)
# ---------------------------------------------------------------
def vectorization_demo():
    # Ratings out of 5 for a few restaurants
    restaurants = ["Domino's", "Honest", "Swati Snacks", "Pizza Hut", "Sasuma"]
    ratings = np.array([4.1, 3.8, 4.6, 3.9, 4.3])

    # --- Version 1: for-loop (slow way) ---
    loop_result = np.empty_like(ratings)
    for i in range(len(ratings)):
        loop_result[i] = ratings[i] * 2

    # --- Version 2: vectorized (fast way) ---
    vector_result = ratings * 2

    print("=== Vectorization: multiply ratings by 2 ===")
    print(f"Original ratings: {ratings}")
    print(f"For-loop result:  {loop_result}")
    print(f"Vectorized result: {vector_result}")
    print(f"Same output? {np.array_equal(loop_result, vector_result)}")
    print()
    for name, r in zip(restaurants, vector_result):
        print(f"  {name}: {r}")
    print()

    # Timing on a large dataset: 1 million ratings
    big_ratings = np.random.uniform(1, 5, 1_000_000)

    start = time.perf_counter()
    out = np.empty_like(big_ratings)
    for i in range(len(big_ratings)):
        out[i] = big_ratings[i] * 2
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(20):
        out = big_ratings * 2
    vec_time = (time.perf_counter() - start) / 20

    print("Timing on 1,000,000 ratings:")
    print(f"  For-loop:   {loop_time:.5f} seconds")
    print(f"  Vectorized: {vec_time:.5f} seconds")
    if vec_time > 0:
        print(f"  Vectorized is about {loop_time / vec_time:.0f}x faster")


if __name__ == "__main__":
    show_memory_usage()
    compare_addition_speed()
    vectorization_demo()
