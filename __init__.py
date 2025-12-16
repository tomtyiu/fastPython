"""
FastMath public API.

This module re-exports performance-optimized math utilities from the
fasterPython subpackage for convenient top-level access.
"""

from .fasterPython.fasterPython import (
    # Scalar math
    fast_sqrt,
    fast_ceil,
    fast_floor,
    fast_factorial,
    fast_log,
    fast_exp,
    fast_sin,

    # Constants
    FAST_PI,
    FAST_E,
    FAST_TAU,
    FAST_INF,
    FAST_NAN,

    # Algorithms
    fib,
    sum_fast,
    compute_squares,

    # Optional accelerators
    sum_numpy,

    # External integrations
    openai_api,
)

__all__ = [
    # Scalar math
    "fast_sqrt",
    "fast_ceil",
    "fast_floor",
    "fast_factorial",
    "fast_log",
    "fast_exp",
    "fast_sin",

    # Constants
    "FAST_PI",
    "FAST_E",
    "FAST_TAU",
    "FAST_INF",
    "FAST_NAN",

    # Algorithms
    "fib",
    "sum_fast",
    "compute_squares",

    # Optional accelerators
    "sum_numpy",

    # Integrations
    "openai_api",
]
