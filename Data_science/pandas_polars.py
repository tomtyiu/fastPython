import pandas as pd
import polars as pl
import time

# Data size
N = 10_000_000

# ---- Pandas ----
df_pandas = pd.DataFrame({
    "a": range(N),
    "b": range(N)
})

start = time.time()
df_pandas["c"] = df_pandas["a"] + df_pandas["b"]
print("Pandas time:", time.time() - start)

# ---- Polars ----
df_polars = pl.DataFrame({
    "a": range(N),
    "b": range(N)
})

start = time.time()
df_polars = df_polars.with_columns(
    (pl.col("a") + pl.col("b")).alias("c")
)
print("Polars time:", time.time() - start)
