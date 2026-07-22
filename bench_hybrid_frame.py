"""Benchmark suite for HybridFrame using pytest-benchmark."""

import numpy as np
import pandas as pd
import pytest

from hybrid_frame import HybridFrame

N = 100_000
CATEGORIES = ["alpha", "beta", "gamma", "delta"]


@pytest.fixture(scope="module")
def large_df():
    np.random.seed(42)
    return pd.DataFrame({
        "key": np.random.choice(CATEGORIES, size=N),
        "val": np.random.randn(N).astype(np.float32),
        "cat": np.random.choice(["x", "y", "z"], size=N),
        "flag": np.random.randint(0, 2, size=N).astype(bool),
    })


@pytest.fixture(scope="module")
def large_hf(large_df):
    return HybridFrame.from_pandas(large_df)


class TestBenchmarks:

    def bench_groupby_agg(self, benchmark, large_hf):
        def run():
            hf = large_hf.copy()
            hf.groupby_agg(["key"], {"val": "sum"})
            return hf.to_pandas()
        result = benchmark(run)
        assert "val_sum" in result.columns
        assert len(result) == len(CATEGORIES)

    def bench_filter(self, benchmark, large_hf):
        def run():
            hf = large_hf.copy()
            hf.filter("val > 0")
            return hf.to_pandas()
        result = benchmark(run)
        assert result.shape[0] > 0

    def bench_sort_limit(self, benchmark, large_hf):
        def run():
            hf = large_hf.copy()
            hf.sort_values("val", ascending=False).limit(10)
            return hf.to_pandas()
        result = benchmark(run)
        assert len(result) == 10

    def bench_one_hot_encode(self, benchmark, large_hf):
        def run():
            hf = large_hf.copy()
            hf.one_hot_encode(columns=["cat"])
            return hf.to_pandas()
        result = benchmark(run)
        assert "cat_x" in result.columns

    def bench_fillna_isna_chain(self, benchmark, large_hf):
        def run():
            hf = large_hf.copy()
            hf.fillna(0)
            return hf.isna().sum().sum()
        result = benchmark(run)
        assert result == 0
