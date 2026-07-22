"""
Memory benchmark for HybridFrame — measures peak and delta memory for every
public method at 10K, 100K, 1M, 10M synthetic rows.

Usage:
    HYBRIDFRAME_MEMORY_TRACE=1 python3 bench_memory.py
"""

import sys
import os
import tracemalloc
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Enable memory tracing before importing HybridFrame
os.environ["HYBRIDFRAME_MEMORY_TRACE"] = "1"
tracemalloc.start()

from hybrid_frame import HybridFrame, _estimate_relation_memory  # noqa: E402

SEED = 42
N_ROWS_LIST = [10_000, 100_000, 1_000_000]
COLS = {
    "int8": lambda n: np.random.randint(-128, 127, n).astype(np.int8),
    "int64": lambda n: np.random.randint(0, 1_000_000, n).astype(np.int64),
    "float64": lambda n: np.random.randn(n).astype(np.float64),
    "float32": lambda n: np.random.randn(n).astype(np.float32),
    "str": lambda n: np.random.choice(
        ["alpha", "beta", "gamma", "delta", "epsilon"], n
    ),
    "bool": lambda n: np.random.choice([True, False], n),
    "datetime": lambda n: pd.date_range("2020-01-01", periods=n, freq="s").values,
}
COL_NAMES = list(COLS.keys())


def make_df(n_rows: int) -> pd.DataFrame:
    np.random.seed(SEED)
    data = {name: gen(n_rows) for name, gen in COLS.items()}
    return pd.DataFrame(data)


def log(label: str, n: int, elapsed: float) -> None:
    print(f"  [{n:>8,}] {label:<45s} {elapsed:>8.3f}s", flush=True)


def run_benchmarks() -> None:
    print("=" * 80)
    print("HybridFrame Memory Benchmark")
    print("=" * 80)
    print(f"{'Rows':>10} {'Operation':<50} {'Time':>10}")
    print("-" * 80)

    for n_rows in N_ROWS_LIST:
        pdf = make_df(n_rows)
        print(f"\n--- N_ROWS = {n_rows:,} ---")

        # 1. from_pandas
        t0 = time.perf_counter()
        hf = HybridFrame.from_pandas(pdf, copy=True)
        log("from_pandas(copy=True)", n_rows, time.perf_counter() - t0)

        # 2. to_duckdb (transition)
        t0 = time.perf_counter()
        hf._to_duckdb()
        log("_to_duckdb (from_df)", n_rows, time.perf_counter() - t0)

        # 3. to_pandas (materialise)
        t0 = time.perf_counter()
        pdf2 = hf.to_pandas(copy=False)
        log("to_pandas (materialise)", n_rows, time.perf_counter() - t0)
        del pdf2

        # 4. filter + materialise
        hf2 = hf.copy()
        t0 = time.perf_counter()
        hf2.filter("int64 > 500000")
        _ = hf2.to_pandas(copy=False)
        log("filter → to_pandas", n_rows, time.perf_counter() - t0)

        # 5. select
        hf3 = hf.copy()
        t0 = time.perf_counter()
        hf3.select(["int64", "float64", "str"])
        _ = hf3.to_pandas(copy=False)
        log("select → to_pandas", n_rows, time.perf_counter() - t0)

        # 6. head
        t0 = time.perf_counter()
        _ = hf.head(100)
        log("head(100)", n_rows, time.perf_counter() - t0)

        # 7. groupby_agg
        hf4 = hf.copy()
        t0 = time.perf_counter()
        hf4.groupby_agg(["str"], {"int64": "sum", "float64": "avg"})
        _ = hf4.to_pandas(copy=False)
        log("groupby_agg → to_pandas", n_rows, time.perf_counter() - t0)

        # 8. join (same-connection)
        left = HybridFrame.from_pandas(
            pd.DataFrame({"k": range(min(1000, n_rows)), "val": np.random.randn(min(1000, n_rows))})
        )._to_duckdb()
        hf5 = HybridFrame.from_pandas(
            pdf[["int64", "float64"]].rename(columns={"int64": "k"}).head(min(1000, n_rows))
        )._to_duckdb()
        t0 = time.perf_counter()
        hf5.join(left, on="k", how="left")
        _ = hf5.to_pandas(copy=False)
        log("join → to_pandas", n_rows, time.perf_counter() - t0)

        # 9. sort + limit
        hf6 = hf.copy()
        t0 = time.perf_counter()
        hf6.sort_values("float64", ascending=False).limit(100)
        _ = hf6.to_pandas(copy=False)
        log("sort+limit → to_pandas", n_rows, time.perf_counter() - t0)

        # 10. fillna (lazy SQL) – numeric columns only
        with_nan = pdf[["int8", "int64", "float64", "float32"]].copy()
        with_nan.loc[::10, "float64"] = None
        hf7 = HybridFrame.from_pandas(with_nan)._to_duckdb()
        t0 = time.perf_counter()
        hf7.fillna(0)
        _ = hf7.to_pandas(copy=False)
        log("fillna(numeric) → to_pandas", n_rows, time.perf_counter() - t0)

        # 11. isna (DuckDB path)
        hf_isna = HybridFrame.from_pandas(pdf)._to_duckdb()
        t0 = time.perf_counter()
        _ = hf_isna.isna()
        log("isna (DuckDB path, materialise full)", n_rows, time.perf_counter() - t0)

        # 12. nunique (DuckDB path)
        hf_nunique = HybridFrame.from_pandas(pdf)._to_duckdb()
        t0 = time.perf_counter()
        _ = hf_nunique.nunique()
        log("nunique (DuckDB path)", n_rows, time.perf_counter() - t0)

        # 13. value_counts (DuckDB path)
        hf_vc = HybridFrame.from_pandas(pdf)._to_duckdb()
        t0 = time.perf_counter()
        _ = hf_vc.value_counts("str")
        log("value_counts(str) (DuckDB path)", n_rows, time.perf_counter() - t0)

        # 14. describe
        t0 = time.perf_counter()
        _ = hf.describe()
        log("describe (SUMMARIZE)", n_rows, time.perf_counter() - t0)

        # 15. fetch_chunked (iterate all) – skip at 1M+ due to overhead
        if n_rows <= 100_000:
            t0 = time.perf_counter()
            total_rows = 0
            for chunk in hf.fetch_chunked(batch_size=4096):
                total_rows += chunk.shape[0]
            log(f"fetch_chunked (batch=4096, {total_rows} rows)", n_rows, time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            total_rows = 0
            for chunk in hf.fetch_chunked(batch_size=65536):
                total_rows += chunk.shape[0]
            log(f"fetch_chunked (batch=65536, {total_rows} rows)", n_rows, time.perf_counter() - t0)

        # 16. copy + to_pandas (double-copy test, from DuckDB)
        t0 = time.perf_counter()
        hf9 = hf.copy()
        _ = hf9.to_pandas(copy=True)
        log("copy → to_pandas(copy=True) (from DuckDB)", n_rows, time.perf_counter() - t0)

        # 17. astype (DuckDB path)
        hf10 = HybridFrame.from_pandas(pdf)._to_duckdb()
        t0 = time.perf_counter()
        hf10.astype({"int8": "BIGINT", "int64": "VARCHAR"})
        _ = hf10.to_pandas(copy=False)
        log("astype (DuckDB path) → to_pandas", n_rows, time.perf_counter() - t0)

        # 18. one_hot_encode (DuckDB path)
        hf11 = HybridFrame.from_pandas(pdf[["str", "int64"]])._to_duckdb()
        t0 = time.perf_counter()
        hf11.one_hot_encode(columns=["str"], drop_first=True)
        _ = hf11.to_pandas(copy=False)
        log("one_hot_encode (DuckDB path) → to_pandas", n_rows, time.perf_counter() - t0)

        # 19. __getitem__ single column (DuckDB path)
        t0 = time.perf_counter()
        _ = hf["int64"]
        log("__getitem__(single col) (DuckDB path)", n_rows, time.perf_counter() - t0)

        # 20. __getitem__ multi column (DuckDB path)
        t0 = time.perf_counter()
        _ = hf[["int64", "float64"]]
        log("__getitem__(multi col) (DuckDB path)", n_rows, time.perf_counter() - t0)

        hf.close()

    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


def estimate_accuracy_check() -> None:
    """Compare _estimate_relation_memory against actual Pandas memory_usage."""
    print("\n--- Estimate Accuracy Check ---")
    for n_rows in N_ROWS_LIST:
        pdf = make_df(n_rows)
        actual_bytes = pdf.memory_usage(deep=True).sum()
        actual_gb = actual_bytes / (1024 ** 3)

        hf = HybridFrame.from_pandas(pdf)._to_duckdb()
        estimated_gb = _estimate_relation_memory(hf._relation)
        ratio = estimated_gb / actual_gb if actual_gb > 0 else 0
        print(
            f"  n={n_rows:>8,}  "
            f"actual={actual_gb:.4f} GB  "
            f"estimated={estimated_gb:.4f} GB  "
            f"ratio={ratio:.2f}x"
        )
        hf.close()


if __name__ == "__main__":
    print("Tracemalloc tracing:", tracemalloc.is_tracing())
    run_benchmarks()
    estimate_accuracy_check()
    tracemalloc.stop()
