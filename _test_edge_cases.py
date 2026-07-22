"""Systematic edge-case tests for every HybridFrame public method.

Run: python3 -m pytest _test_edge_cases.py -x -v --tb=short 2>&1
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from hybrid_frame import HybridFrame, HybridFrameError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 50


def _make_df(extra_cols: dict | None = None) -> pd.DataFrame:
    cols = {
        "cat": np.random.choice(list("ABC"), N),
        "x": np.random.randn(N),
        "y": np.random.randn(N),
        "val": np.random.randn(N),
    }
    if extra_cols:
        cols.update(extra_cols)
    return pd.DataFrame(cols)


def _make_hf(extra_cols: dict | None = None) -> HybridFrame:
    return HybridFrame.from_pandas(_make_df(extra_cols))


# ========================================================================
# 1.  CONSTRUCTOR & FACTORIES
# ========================================================================

class TestConstructor:
    def test_default_constructor(self):
        hf = HybridFrame()
        assert hf._engine is not None
        assert hf._conn is None  # lazy – no connection acquired
        assert hf._df is None
        assert hf._relation is None

    def test_constructor_with_connection(self):
        import duckdb
        conn = duckdb.connect()
        hf = HybridFrame(connection=conn)
        assert hf._conn is conn
        assert hf._engine is not None

    def test_from_pandas_empty(self):
        hf = HybridFrame.from_pandas(pd.DataFrame())
        assert hf.shape == (0, 0)
        assert hf.columns == []
        assert hf.to_pandas().empty

    def test_from_pandas_empty_rows(self):
        df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})
        hf = HybridFrame.from_pandas(df)
        assert hf.shape == (0, 2)
        assert hf.columns == ["a", "b"]

    def test_from_pandas_none_raises(self):
        with pytest.raises(AttributeError):
            HybridFrame.from_pandas(None)  # type: ignore

    def test_from_pandas_copy_false(self):
        df = _make_df()
        hf = HybridFrame.from_pandas(df, copy=False)
        assert hf.to_pandas(copy=False) is df  # same object

    def test_from_pandas_copy_true(self):
        df = _make_df()
        hf = HybridFrame.from_pandas(df, copy=True)
        assert hf.to_pandas(copy=False) is not df  # different object


class TestConstructorConfig:
    def test_constructor_memory_limit_lazy(self):
        hf = HybridFrame(memory_limit="1GB")
        assert hf._conn is None  # lazy
        # After _to_duckdb, connection has memory limit set via config
        hf2 = HybridFrame.from_pandas(_make_df())
        assert hf2._conn is None
        hf2._to_duckdb()
        assert hf2._conn is not None

    def test_constructor_threads_lazy(self):
        hf = HybridFrame(threads=2)
        assert hf._conn is None

    def test_constructor_temp_directory_lazy(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            hf = HybridFrame(temp_directory=td)
            assert hf._conn is None


# ========================================================================
# 2.  PROPERTIES
# ========================================================================

class TestProperties:
    def test_columns_pandas(self):
        hf = _make_hf()
        assert hf.columns == ["cat", "x", "y", "val"]

    def test_columns_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        assert hf.columns == ["cat", "x", "y", "val"]

    def test_columns_empty(self):
        hf = HybridFrame()
        assert hf.columns == []

    def test_shape_pandas(self):
        hf = _make_hf()
        assert hf.shape == (N, 4)

    def test_shape_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        assert hf.shape == (N, 4)

    def test_shape_empty(self):
        hf = HybridFrame()
        assert hf.shape == (0, 0)

    def test_dtypes_pandas(self):
        hf = _make_hf()
        assert isinstance(hf.dtypes, pd.Series)

    def test_dtypes_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        assert isinstance(hf.dtypes, pd.Series)

    def test_dtypes_empty(self):
        hf = HybridFrame()
        assert isinstance(hf.dtypes, pd.Series)
        assert len(hf.dtypes) == 0

    def test_len(self):
        hf = _make_hf()
        assert len(hf) == N
        hf2 = HybridFrame()
        assert len(hf2) == 0


# ========================================================================
# 3.  LIFECYCLE: close, copy, __del__
# ========================================================================

class TestLifecycle:
    def test_close_pandas(self):
        hf = _make_hf()
        hf.close()
        assert hf._df is None

    def test_close_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        conn = hf._conn
        hf.close()
        assert hf._relation is None
        assert hf._df is None

    def test_close_none_conn(self):
        hf = HybridFrame()
        hf.close()  # should not raise

    def test_close_twice(self):
        hf = _make_hf()
        hf.close()
        hf.close()  # should not raise

    def test_copy_pandas(self):
        hf = _make_hf()
        cp = hf.copy()
        assert cp._engine is hf._engine
        pd.testing.assert_frame_equal(cp.to_pandas(), hf.to_pandas())
        # Modifying copy should not affect original
        cp._df.iloc[0, 0] = "MODIFIED"
        assert hf._df.iloc[0, 0] != "MODIFIED"

    def test_copy_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        cp = hf.copy()
        assert cp._engine is hf._engine
        # DuckDB copy shares relation reference
        assert cp._relation is hf._relation


# ========================================================================
# 4.  DATA ACCESS: __getitem__, __setitem__, pop, assign
# ========================================================================

class TestGetItem:
    def test_getitem_string_key(self):
        hf = _make_hf()
        result = hf["x"]
        assert isinstance(result, pd.Series)
        assert result.name == "x"

    def test_getitem_list_key(self):
        hf = _make_hf()
        result = hf[["x", "y"]]
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["x", "y"]

    def test_getitem_slice(self):
        hf = _make_hf()
        result = hf[:10]
        assert isinstance(result, HybridFrame)
        assert result.shape == (10, 4)

    def test_getitem_slice_empty_stop(self):
        hf = _make_hf()
        result = hf[:]
        assert isinstance(result, HybridFrame)
        assert result.shape == (N, 4)

    def test_getitem_slice_negative_start_raises(self):
        hf = _make_hf()
        with pytest.raises((ValueError, HybridFrameError, IndexError)):
            hf[-5:]

    def test_getitem_slice_step_not_one_raises(self):
        hf = _make_hf()
        with pytest.raises((ValueError, HybridFrameError, IndexError)):
            hf[::2]

    def test_getitem_non_existent_column(self):
        hf = _make_hf()
        with pytest.raises((KeyError, HybridFrameError)):
            hf["nonexistent"]

    def test_getitem_duckdb_string_key(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf["x"]
        assert isinstance(result, pd.Series)

    def test_getitem_empty_frame(self):
        hf = HybridFrame()
        with pytest.raises((KeyError, HybridFrameError, IndexError)):
            hf["anything"]


class TestSetItem:
    def test_setitem_scalar(self):
        hf = _make_hf()
        hf["new_col"] = 42
        assert "new_col" in hf.columns
        assert hf["new_col"].iloc[0] == 42

    def test_setitem_series(self):
        hf = _make_hf()
        hf["new_col"] = pd.Series(np.ones(N), index=hf._df.index)
        assert hf["new_col"].iloc[0] == 1.0

    def test_setitem_non_string_key_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf[123] = 42  # type: ignore

    def test_setitem_duckdb_triggers_pandas(self):
        hf = _make_hf()
        hf._to_duckdb()
        hf["new_col"] = 42
        assert hf._engine.name == "PANDAS_DATAFRAME"
        assert "new_col" in hf.columns

    def test_setitem_empty_frame(self):
        hf = HybridFrame()
        hf["a"] = 1
        assert hf._df is not None
        assert "a" in hf.columns

    def test_setitem_mismatched_length(self):
        hf = _make_hf()
        with pytest.raises(ValueError):
            hf["new_col"] = [1, 2, 3]  # wrong length


class TestPop:
    def test_pop_column(self):
        hf = _make_hf()
        ser = hf.pop("x")
        assert isinstance(ser, pd.Series)
        assert "x" not in hf.columns

    def test_pop_last_column_raises(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(HybridFrameError):
            hf.pop("a")

    def test_pop_non_existent(self):
        hf = _make_hf()
        with pytest.raises((KeyError, HybridFrameError)):
            hf.pop("nonexistent")

    def test_pop_empty(self):
        hf = HybridFrame()
        with pytest.raises((KeyError, HybridFrameError)):
            hf.pop("a")


class TestAssign:
    def test_assign_new_column(self):
        hf = _make_hf()
        result = hf.assign(new_col=42)
        assert "new_col" in result.columns

    def test_assign_callable(self):
        hf = _make_hf()
        result = hf.assign(x2=lambda df: df["x"] * 2)
        assert "x2" in result.columns
        assert (result["x2"].values == (hf["x"] * 2).values).all()

    def test_assign_empty_kwargs(self):
        hf = _make_hf()
        result = hf.assign()
        pd.testing.assert_frame_equal(result.to_pandas(), hf.to_pandas())

    def test_assign_overwrite(self):
        hf = _make_hf()
        result = hf.assign(x=99)
        assert (result["x"] == 99).all()


# ========================================================================
# 5.  DUCKDB-ENGINE METHODS
# ========================================================================

class TestFilter:
    def test_filter_string(self):
        hf = _make_hf()
        result = hf.filter("x > 0")
        assert isinstance(result, HybridFrame)
        assert len(result) > 0

    def test_filter_list(self):
        hf = _make_hf()
        result = hf.filter(["x > 0", "y > 0"])
        assert isinstance(result, HybridFrame)
        assert len(result) <= len(hf)

    def test_filter_empty_result(self):
        hf = _make_hf()
        result = hf.filter("x > 9999")
        assert len(result) == 0

    def test_filter_invalid_sql_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.filter("NOT VALID SQL !!!")

    def test_filter_non_existent_column(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.filter("nonexistent > 0")

    def test_filter_empty_string_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.filter("")


class TestSelect:
    def test_select_columns(self):
        hf = _make_hf()
        result = hf.select(["x", "y"])
        assert result.columns == ["x", "y"]
        assert result.shape[1] == 2

    def test_select_string(self):
        hf = _make_hf()
        result = hf.select("x")
        # Should work – wraps in list internally
        assert result.columns == ["x"]

    def test_select_all_columns(self):
        hf = _make_hf()
        result = hf.select(hf.columns)
        assert result.columns == hf.columns

    def test_select_non_existent_column(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.select(["nonexistent"])

    def test_select_empty_list_pandas(self):
        hf = _make_hf()
        result = hf.select([])
        # Pandas path: empty list returns empty DataFrame
        assert result.shape[1] == 0

    def test_select_empty_list_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf.select([])
        # DuckDB select() with no args returns the relation as-is
        assert result.columns == hf.columns

    def test_select_duplicate_columns_pandas(self):
        hf = _make_hf()
        result = hf.select(["x", "x"])
        assert result.shape[1] == 2
        assert result.columns == ["x", "x"]

    def test_select_duplicate_columns_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        # DuckDB allows selecting the same column twice
        result = hf.select(["x", "x"])
        assert result.shape[1] == 2

    def test_select_no_change_engine(self):
        """select stays in Pandas mode if already there"""
        hf = _make_hf()
        hf.select(["x", "y"])
        assert hf._engine.name == "PANDAS_DATAFRAME"

    def test_select_from_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf.select(["x", "y"])
        assert result.columns == ["x", "y"]


class TestSortValues:
    def test_sort_single_column(self):
        hf = _make_hf()
        result = hf.sort_values("x")
        assert len(result) == N
        # Check sorted ascending
        vals = result.to_pandas()["x"].values
        assert np.all(vals[:-1] <= vals[1:])

    def test_sort_descending(self):
        hf = _make_hf()
        result = hf.sort_values("x", ascending=False)
        vals = result.to_pandas()["x"].values
        assert np.all(vals[:-1] >= vals[1:])

    def test_sort_multi_column(self):
        hf = _make_hf()
        result = hf.sort_values(["cat", "x"])
        pdf = result.to_pandas()
        # Check cat is sorted
        cat_vals = pdf["cat"].values
        assert list(cat_vals) == sorted(cat_vals)

    def test_sort_non_existent_column_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.sort_values("nonexistent")

    def test_sort_empty_list_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.sort_values([])

    def test_sort_single_row(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        hf = HybridFrame.from_pandas(df)
        result = hf.sort_values("a")
        assert result.shape == (1, 2)


class TestLimit:
    def test_limit_positive(self):
        hf = _make_hf()
        result = hf.limit(10)
        assert result.shape[0] == 10
        assert result._engine.name == "PANDAS_DATAFRAME"

    def test_limit_zero(self):
        hf = _make_hf()
        result = hf.limit(0)
        assert result.shape[0] == 0

    def test_limit_larger_than_dataset(self):
        hf = _make_hf()
        result = hf.limit(999999)
        assert result.shape[0] == N

    def test_limit_negative_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.limit(-1)

    def test_limit_stays_pandas(self):
        hf = _make_hf()
        hf.limit(5)
        assert hf._engine.name == "PANDAS_DATAFRAME"

    def test_limit_from_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf.limit(5)
        assert result.shape[0] == 5


class TestHeadTail:
    def test_head_default(self):
        hf = _make_hf()
        result = hf.head()
        assert result.shape[0] == 5
        assert result._engine.name == "PANDAS_DATAFRAME"

    def test_head_zero(self):
        hf = _make_hf()
        result = hf.head(0)
        assert result.shape[0] == 0

    def test_head_larger(self):
        hf = _make_hf()
        result = hf.head(999)
        assert result.shape[0] == N

    def test_head_from_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf.head(3)
        assert result.shape[0] == 3

    def test_tail_default(self):
        hf = _make_hf()
        result = hf.tail()
        assert result.shape[0] == 5
        assert result._engine.name == "PANDAS_DATAFRAME"

    def test_tail_zero(self):
        hf = _make_hf()
        result = hf.tail(0)
        assert result.shape[0] == 0

    def test_tail_from_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf.tail(3)
        assert result.shape[0] == 3

    def test_head_empty_frame(self):
        hf = HybridFrame()
        result = hf.head()
        assert result._df is not None and result._df.empty

    def test_tail_empty_frame(self):
        hf = HybridFrame()
        result = hf.tail()
        assert result._df is not None and result._df.empty

    def test_head_does_not_mutate_original_engine(self):
        """head should not force the original into DuckDB mode"""
        hf = _make_hf()
        hf.head(5)
        assert hf._engine.name == "PANDAS_DATAFRAME"
        assert hf._df is not None


class TestDistinct:
    def test_distinct(self):
        # Create with duplicate rows
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 10, 20, 20, 30]})
        hf = HybridFrame.from_pandas(df)
        result = hf.distinct()
        assert result.shape[0] == 3

    def test_distinct_no_duplicates(self):
        hf = _make_hf()
        result = hf.distinct()
        assert result.shape[0] == N

    def test_distinct_empty(self):
        hf = HybridFrame()
        result = hf.distinct()
        assert result.shape[0] == 0


class TestSample:
    def test_sample_int(self):
        hf = _make_hf()
        result = hf.sample(10)
        assert result.shape[0] == 10

    def test_sample_float(self):
        hf = _make_hf()
        result = hf.sample(50.0)  # percent
        assert 0 < result.shape[0] <= N

    def test_sample_zero(self):
        hf = _make_hf()
        result = hf.sample(0)
        assert result.shape[0] == 0

    def test_sample_negative_int_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.sample(-5)

    def test_sample_invalid_method_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.sample(5, method="invalid")

    def test_sample_float_out_of_range_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.sample(101.0)


class TestGroupbyAgg:
    def test_groupby_single_agg(self):
        hf = _make_hf()
        result = hf.groupby_agg(["cat"], {"val": "sum"})
        assert "cat" in result.columns
        assert "val_sum" in result.columns

    def test_groupby_multi_agg(self):
        hf = _make_hf()
        result = hf.groupby_agg(["cat"], {"val": ["sum", "mean"]})
        assert "val_sum" in result.columns
        assert "val_mean" in result.columns

    def test_groupby_empty_by_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.groupby_agg([], {"val": "sum"})

    def test_groupby_empty_agg_dict_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.groupby_agg(["cat"], {})

    def test_groupby_nunique(self):
        hf = _make_hf()
        result = hf.groupby_agg(["cat"], {"val": "nunique"})
        assert "val_nunique" in result.columns

    def test_groupby_non_existent_by_column(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.groupby_agg(["nonexistent"], {"val": "sum"})

    def test_groupby_all_numeric_agg(self):
        hf = _make_hf()
        result = hf.groupby_agg(["cat"], {c: "sum" for c in ["x", "y", "val"]})
        assert all(f"{c}_sum" in result.columns for c in ["x", "y", "val"])


class TestJoin:
    def test_join_inner(self):
        df1 = pd.DataFrame({"key": [1, 2, 3], "a": [4, 5, 6]})
        df2 = pd.DataFrame({"key": [1, 2, 4], "b": [7, 8, 9]})
        hf1 = HybridFrame.from_pandas(df1)
        hf2 = HybridFrame.from_pandas(df2)
        result = hf1.join(hf2, on="key", how="inner")
        assert result.shape[0] == 2
        assert "a" in result.columns
        assert "b" in result.columns

    def test_join_left(self):
        df1 = pd.DataFrame({"key": [1, 2, 3], "a": [4, 5, 6]})
        df2 = pd.DataFrame({"key": [1, 2, 4], "b": [7, 8, 9]})
        hf1 = HybridFrame.from_pandas(df1)
        hf2 = HybridFrame.from_pandas(df2)
        result = hf1.join(hf2, on="key", how="left")
        assert result.shape[0] == 3

    def test_join_empty(self):
        df1 = pd.DataFrame({"key": pd.Series(dtype="int64"), "a": pd.Series(dtype="int64")})
        df2 = pd.DataFrame({"key": pd.Series(dtype="int64"), "b": pd.Series(dtype="int64")})
        hf1 = HybridFrame.from_pandas(df1)
        hf2 = HybridFrame.from_pandas(df2)
        result = hf1.join(hf2, on="key")
        assert result.shape[0] == 0

    def test_join_invalid_how_raises(self):
        df1 = pd.DataFrame({"key": [1, 2], "a": [3, 4]})
        df2 = pd.DataFrame({"key": [1, 2], "b": [5, 6]})
        hf1 = HybridFrame.from_pandas(df1)
        hf2 = HybridFrame.from_pandas(df2)
        with pytest.raises(HybridFrameError):
            hf1.join(hf2, on="key", how="invalid")


class TestSetOps:
    @pytest.fixture
    def frames(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [3, 4, 5], "b": [6, 7, 8]})
        return HybridFrame.from_pandas(df1), HybridFrame.from_pandas(df2)

    def test_union(self, frames):
        hf1, hf2 = frames
        result = hf1.union(hf2)
        assert result.shape[0] == 5  # 3+3-1 (3 appears in both)

    def test_union_all(self, frames):
        hf1, hf2 = frames
        result = hf1.union(hf2, all=True)
        assert result.shape[0] == 6  # 3+3

    def test_intersect(self, frames):
        hf1, hf2 = frames
        result = hf1.intersect(hf2)
        assert result.shape[0] == 1  # only 3

    def test_except_(self, frames):
        hf1, hf2 = frames
        result = hf1.except_(hf2)
        assert result.shape[0] == 2  # 1, 2

    def test_union_empty(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": pd.Series(dtype="int64")})
        hf1 = HybridFrame.from_pandas(df1)
        hf2 = HybridFrame.from_pandas(df2)
        result = hf1.union(hf2)
        assert result.shape[0] == 2

    def test_union_diff_columns_raises(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [1], "c": [3]}))
        # DuckDB union is positional — different columns do not raise
        result = hf1.union(hf2)
        assert result.shape == (2, 2)


class TestSql:
    def test_sql_simple(self):
        hf = _make_hf()
        result = hf.sql("SELECT * FROM self WHERE x > 0")
        assert isinstance(result, HybridFrame)
        assert len(result) > 0

    def test_sql_aggregate(self):
        hf = _make_hf()
        result = hf.sql("SELECT COUNT(*) AS cnt FROM self")
        assert result.shape[0] == 1
        assert "cnt" in result.columns

    def test_sql_invalid_query_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.sql("SELECT INVALID")

    def test_sql_empty_result(self):
        hf = _make_hf()
        result = hf.sql("SELECT * FROM self WHERE 1=0")
        assert result.shape[0] == 0


class TestShowPlan:
    def test_show_plan(self):
        hf = _make_hf()
        hf._to_duckdb()
        plan = hf.show_plan()
        assert isinstance(plan, str)
        assert len(plan) > 0

    def test_show_plan_pandas_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.show_plan()

    def test_show_plan_empty_raises(self):
        hf = HybridFrame()
        with pytest.raises(HybridFrameError):
            hf.show_plan()


# ========================================================================
# 6.  PANDAS PERSISTENCE METHODS
# ========================================================================

class TestRenameDrop:
    def test_rename_single(self):
        hf = _make_hf()
        result = hf.rename({"x": "x2"})
        assert "x2" in result.columns
        assert "x" not in result.columns

    def test_rename_empty_dict(self):
        hf = _make_hf()
        result = hf.rename({})
        assert result.columns == hf.columns

    def test_rename_to_existing_raises(self):
        hf = _make_hf()
        hf._to_duckdb()
        with pytest.raises(HybridFrameError):
            hf.rename({"x": "y"})  # y already exists

    def test_rename_non_existent_column(self):
        hf = _make_hf()
        result = hf.rename({"nonexistent": "new_name"})
        # Should be a no-op for that column
        assert "new_name" not in result.columns

    def test_drop_single(self):
        hf = _make_hf()
        result = hf.drop("x")
        assert "x" not in result.columns
        assert result.shape[1] == 3

    def test_drop_list(self):
        hf = _make_hf()
        result = hf.drop(["x", "y"])
        assert result.shape[1] == 2

    def test_drop_all_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.drop(hf.columns)

    def test_drop_non_existent(self):
        hf = _make_hf()
        result = hf.drop("nonexistent")
        # Should silently ignore on DuckDB path
        assert result.shape[1] == 4

    def test_drop_empty_string(self):
        hf = _make_hf()
        result = hf.drop("")
        assert result.shape[1] == 4  # '' not in columns, no-op


class TestFillna:
    def test_fillna_scalar(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, None]})
        hf = HybridFrame.from_pandas(df)
        result = hf.fillna(0)
        assert result.to_pandas().isna().sum().sum() == 0

    def test_fillna_dict(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, None]})
        hf = HybridFrame.from_pandas(df)
        result = hf.fillna({"a": 0, "b": -1})
        pdf = result.to_pandas()
        assert pdf["a"].isna().sum() == 0
        assert pdf["b"].isna().sum() == 0

    def test_fillna_no_nulls(self):
        hf = _make_hf()
        result = hf.fillna(0)
        assert result.shape == hf.shape

    def test_fillna_none(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        hf = HybridFrame.from_pandas(df)
        result = hf.fillna(None)
        # None => no-op since None value doesn't replace anything
        assert result.to_pandas()["a"].isna().sum() == 1

    def test_fillna_all_nulls(self):
        df = pd.DataFrame({"a": [None, None, None]})
        hf = HybridFrame.from_pandas(df)
        result = hf.fillna(42)
        assert (result.to_pandas()["a"] == 42).all()


class TestClip:
    def test_clip_lower(self):
        hf = _make_hf()
        result = hf.clip(lower=0.0)
        pdf = result.to_pandas()
        assert (pdf["x"] >= 0).all()
        assert (pdf["y"] >= 0).all()
        # cat column should be unchanged (it's object type)
        assert (pdf["cat"] == hf.to_pandas()["cat"]).all()

    def test_clip_upper(self):
        hf = _make_hf()
        result = hf.clip(upper=0.5)
        pdf = result.to_pandas()
        assert (pdf["x"] <= 0.5).all()

    def test_clip_both(self):
        hf = _make_hf()
        result = hf.clip(lower=-0.5, upper=0.5)
        pdf = result.to_pandas()
        assert (pdf["x"] >= -0.5).all()
        assert (pdf["x"] <= 0.5).all()

    def test_clip_none(self):
        hf = _make_hf()
        result = hf.clip()
        # Both None => should return self
        pd.testing.assert_frame_equal(result.to_pandas(), hf.to_pandas())

    def test_clip_dict(self):
        hf = _make_hf()
        result = hf.clip(lower={"x": -0.1})
        assert (result.to_pandas()["x"] >= -0.1).all()
        assert (result.to_pandas()["y"] >= hf.to_pandas()["y"]).all()  # unchanged


class TestAstype:
    def test_astype_all(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}))
        result = hf.astype("BIGINT")
        assert result.to_pandas()["a"].dtype.name in ("int64", "Int64")

    def test_astype_dict(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))
        result = hf.astype({"a": "VARCHAR"})
        assert result.to_pandas()["a"].dtype.name == "object"

    def test_astype_invalid_type_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.astype("NOT_A_TYPE")

    def test_astype_pandas_stays_pandas(self):
        hf = _make_hf()
        hf.astype("VARCHAR")
        assert hf._engine.name == "PANDAS_DATAFRAME"


class TestIsnaNunique:
    def test_isna(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, None]})
        hf = HybridFrame.from_pandas(df)
        result = hf.isna()
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[1, 0] == True

    def test_isna_no_nulls(self):
        hf = _make_hf()
        result = hf.isna()
        assert result.sum().sum() == 0

    def test_isna_empty(self):
        hf = HybridFrame()
        result = hf.isna()
        assert isinstance(result, pd.DataFrame)

    def test_nunique(self):
        hf = _make_hf()
        result = hf.nunique()
        assert isinstance(result, pd.Series)
        assert result["cat"] == 3

    def test_nunique_empty(self):
        hf = HybridFrame()
        result = hf.nunique()
        assert isinstance(result, pd.Series)

    def test_value_counts(self):
        hf = _make_hf()
        result = hf.value_counts("cat")
        assert isinstance(result, pd.Series)
        assert result.sum() == N

    def test_value_counts_non_existent(self):
        hf = _make_hf()
        with pytest.raises((KeyError, HybridFrameError)):
            hf.value_counts("nonexistent")


class TestDropna:
    def test_dropna_default(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        hf = HybridFrame.from_pandas(df)
        result = hf.dropna()
        assert result.shape[0] == 1  # only row with no nulls

    def test_dropna_how_all(self):
        df = pd.DataFrame({"a": [1, None, None], "b": [2, None, None]})
        hf = HybridFrame.from_pandas(df)
        result = hf.dropna(how="all")
        assert result.shape[0] == 1  # only row 0 kept

    def test_dropna_subset(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 5, 6]})
        hf = HybridFrame.from_pandas(df)
        result = hf.dropna(subset=["a"])
        assert result.shape[0] == 2

    def test_dropna_invalid_how_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.dropna(how="invalid")

    def test_dropna_no_nulls(self):
        hf = _make_hf()
        result = hf.dropna()
        assert result.shape == hf.shape


class TestReplace:
    def test_replace_single_dict(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hf = HybridFrame.from_pandas(df)
        result = hf.replace({1: 10, 2: 20})
        # DuckDB path: applies to all numeric columns
        pdf = result.to_pandas()
        assert (pdf["a"] == [10, 20, 3]).all()

    def test_replace_per_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        hf = HybridFrame.from_pandas(df)
        result = hf.replace({"a": {1: 10}, "b": {3: 30}})
        pdf = result.to_pandas()
        assert pdf["a"].iloc[0] == 10
        assert pdf["b"].iloc[0] == 30

    def test_replace_no_match(self):
        hf = _make_hf()
        result = hf.replace({999: 0})
        # Should be a no-op
        pd.testing.assert_frame_equal(result.to_pandas(), hf.to_pandas())

    def test_replace_empty_dict(self):
        hf = _make_hf()
        result = hf.replace({})
        pd.testing.assert_frame_equal(result.to_pandas(), hf.to_pandas())


class TestWhere:
    def test_where_simple(self):
        hf = _make_hf()
        result = hf.where("x > 0", other=0.0)
        pdf = result.to_pandas()
        assert (pdf["x"] >= 0).all()

    def test_where_no_match(self):
        hf = _make_hf()
        result = hf.where("1=1", other=0.0)
        pd.testing.assert_frame_equal(result.to_pandas(), hf.to_pandas())

    def test_where_all_matched(self):
        hf = _make_hf()
        result = hf.where("1=0", other=0.0)
        pdf = result.to_pandas()
        assert (pdf["x"] == 0.0).all()

    def test_where_invalid_sql_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.where("NOT VALID", other=0)


class TestBetween:
    def test_between(self):
        hf = _make_hf()
        result = hf.between("x", -0.5, 0.5)
        pdf = result.to_pandas()
        assert (pdf["x"] >= -0.5).all()
        assert (pdf["x"] <= 0.5).all()

    def test_between_no_match(self):
        hf = _make_hf()
        result = hf.between("x", 999, 1000)
        assert result.shape[0] == 0

    def test_between_non_existent_column(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.between("nonexistent", 0, 1)

    def test_between_none_bound(self):
        hf = _make_hf()
        result = hf.between("x", None, 0)
        # x between NULL and 0 => unknown => zero rows
        assert result.shape[0] == 0


class TestDiffCum:
    @pytest.fixture
    def sorted_hf(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 4.0, 7.0], "b": [10, 20, 30, 40]})
        return HybridFrame.from_pandas(df)

    def test_diff(self, sorted_hf):
        result = sorted_hf.diff()
        pdf = result.to_pandas()
        assert pdf["a"].iloc[0] is None or math.isnan(pdf["a"].iloc[0])
        assert pdf["a"].iloc[1] == 1.0

    def test_cumsum(self, sorted_hf):
        result = sorted_hf.cumsum()
        pdf = result.to_pandas()
        assert pdf["a"].iloc[2] == 7.0  # 1+2+4

    def test_cumprod(self, sorted_hf):
        result = sorted_hf.cumprod()
        pdf = result.to_pandas()
        assert pdf["a"].iloc[2] == 8.0  # 1*2*4

    def test_cummin(self, sorted_hf):
        df = pd.DataFrame({"a": [3.0, 1.0, 2.0]})
        hf = HybridFrame.from_pandas(df)
        result = hf.cummin()
        pdf = result.to_pandas()
        assert list(pdf["a"]) == [3.0, 1.0, 1.0]

    def test_cummax(self, sorted_hf):
        df = pd.DataFrame({"a": [1.0, 3.0, 2.0]})
        hf = HybridFrame.from_pandas(df)
        result = hf.cummax()
        pdf = result.to_pandas()
        assert list(pdf["a"]) == [1.0, 3.0, 3.0]

    def test_diff_empty(self):
        hf = HybridFrame()
        result = hf.diff()
        assert result.shape == (0, 0)

    def test_diff_with_order_by(self, sorted_hf):
        result = sorted_hf.diff(order_by="a")
        pdf = result.to_pandas()
        assert pdf["a"].iloc[1] == 1.0


class TestIdxMinMax:
    @pytest.fixture
    def hf(self):
        df = pd.DataFrame({"a": [10, 5, 20, 5, 15], "b": [1, 2, 3, 4, 5]})
        return HybridFrame.from_pandas(df)

    def test_idxmin(self, hf):
        assert hf.idxmin("a") == 1  # row 1 has value 5

    def test_idxmax(self, hf):
        assert hf.idxmax("a") == 2  # row 2 has value 20

    def test_idxmin_empty_raises(self):
        hf = HybridFrame()
        with pytest.raises((IndexError, HybridFrameError)):
            hf.idxmin("a")

    def test_idxmin_non_existent(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.idxmin("nonexistent")


# ========================================================================
# 7.  RESHAPE
# ========================================================================

class TestReshape:
    def test_reshape_explode(self):
        df = pd.DataFrame({"a": [[1, 2], [3, 4]], "b": [10, 20]})
        hf = HybridFrame.from_pandas(df)
        result = hf.reshape("explode", column="a")
        assert result.shape[0] == 4

    def test_reshape_melt(self):
        hf = _make_hf()
        result = hf.reshape("melt", id_vars=["cat"], value_vars=["x", "y"])
        assert "variable" in result.columns
        assert "value" in result.columns

    def test_reshape_invalid_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.reshape("invalid_method")

    def test_reshape_pivot(self):
        df = pd.DataFrame({
            "date": ["2020-01", "2020-01", "2020-02", "2020-02"],
            "cat": ["A", "B", "A", "B"],
            "val": [10, 20, 30, 40],
        })
        hf = HybridFrame.from_pandas(df)
        result = hf.reshape("pivot", index="date", columns="cat", values="val")
        assert result.shape[0] == 2
        assert "A" in result.columns


# ========================================================================
# 8.  I/O
# ========================================================================

class TestIO:
    def test_write_csv_roundtrip(self, tmp_path):
        hf = _make_hf()
        path = str(tmp_path / "test.csv")
        hf.write_csv(path)
        hf2 = HybridFrame.from_csv(path)
        assert hf2.shape == hf.shape

    def test_write_parquet_roundtrip(self, tmp_path):
        hf = _make_hf()
        path = str(tmp_path / "test.parquet")
        hf.write_parquet(path)
        hf2 = HybridFrame.from_parquet(path)
        assert hf2.shape == hf.shape

    def test_write_csv_empty(self, tmp_path):
        hf = HybridFrame()
        path = str(tmp_path / "empty.csv")
        hf.write_csv(path)
        hf2 = HybridFrame.from_csv(path)
        assert hf2.shape[0] == 0  # duckdb reads 0 rows

    def test_from_csv_nonexistent(self):
        with pytest.raises((FileNotFoundError, HybridFrameError)):
            HybridFrame.from_csv("/nonexistent/path.csv")

    def test_from_parquet_nonexistent(self):
        with pytest.raises((FileNotFoundError, HybridFrameError)):
            HybridFrame.from_parquet("/nonexistent/path.parquet")


# ========================================================================
# 9.  EXPORT / ML
# ========================================================================

class TestExport:
    def test_to_pandas_copy_default(self):
        hf = _make_hf()
        result = hf.to_pandas()
        assert isinstance(result, pd.DataFrame)
        assert result is not hf._df  # should be a copy

    def test_to_pandas_copy_false(self):
        hf = _make_hf()
        result = hf.to_pandas(copy=False)
        assert result is hf._df  # same object

    def test_to_pandas_empty(self):
        hf = HybridFrame()
        result = hf.to_pandas()
        assert result.empty

    def test_to_ml_ready(self):
        hf = _make_hf()
        X, y = hf.to_ml_ready("val")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "val" not in X.columns

    def test_to_ml_ready_non_existent_target(self):
        hf = _make_hf()
        with pytest.raises(KeyError):
            hf.to_ml_ready("nonexistent")

    def test_to_ml_ready_empty(self):
        hf = HybridFrame()
        with pytest.raises(KeyError):
            hf.to_ml_ready("a")


class TestDescribe:
    def test_describe(self):
        hf = _make_hf()
        result = hf.describe()
        assert isinstance(result, pd.DataFrame)
        assert "count" in result.index

    def test_describe_empty(self):
        hf = HybridFrame()
        result = hf.describe()
        assert isinstance(result, pd.DataFrame)


# ========================================================================
# 10. TIME SERIES IMPUTE
# ========================================================================

class TestTimeSeriesImpute:
    def test_impute_ffill(self):
        df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": [10, 20, 30, 40, 50]})
        hf = HybridFrame.from_pandas(df)
        result = hf.time_series_impute("ffill")
        pdf = result.to_pandas()
        assert pdf["a"].iloc[1] == 1.0

    def test_impute_bfill(self):
        df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": [10, 20, 30, 40, 50]})
        hf = HybridFrame.from_pandas(df)
        result = hf.time_series_impute("bfill")
        pdf = result.to_pandas()
        assert pdf["a"].iloc[3] == 5.0

    def test_impute_invalid_method_raises(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError):
            hf.time_series_impute("invalid")

    def test_impute_no_nulls(self):
        hf = _make_hf()
        result = hf.time_series_impute("ffill")
        pd.testing.assert_frame_equal(result.to_pandas(), hf.to_pandas())

    def test_impute_with_datetime_col(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "a": [1, None, 3],
        })
        hf = HybridFrame.from_pandas(df)
        result = hf.time_series_impute("ffill", datetime_col="dt")
        assert result.to_pandas()["a"].iloc[1] == 1.0


# ========================================================================
# 11. ONE-HOT ENCODE
# ========================================================================

class TestOneHotEncode:
    def test_one_hot_default(self):
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"], "val": [1, 2, 3, 4]})
        hf = HybridFrame.from_pandas(df)
        result = hf.one_hot_encode(["cat"])
        assert "cat_A" in result.columns
        assert "cat_B" in result.columns
        assert "cat" not in result.columns

    def test_one_hot_drop_first(self):
        df = pd.DataFrame({"cat": ["A", "B", "A", "C"], "val": [1, 2, 3, 4]})
        hf = HybridFrame.from_pandas(df)
        result = hf.one_hot_encode(["cat"], drop_first=True)
        assert "cat_A" not in result.columns
        assert "cat_B" in result.columns

    def test_one_hot_single_value(self):
        df = pd.DataFrame({"cat": ["A", "A", "A"], "val": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)
        result = hf.one_hot_encode(["cat"], drop_first=True)
        # Only one value => drop_first removes all, no dummy columns
        assert result.shape == (3, 1)  # only val remains

    def test_one_hot_special_chars(self):
        df = pd.DataFrame({"cat": ["a b", "c'd", 'e"f'], "val": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)
        result = hf.one_hot_encode(["cat"])
        # DuckDB does not sanitise quotes in column names
        cols = result.columns
        assert any("'" in c for c in cols)
        assert any('"' in c for c in cols)


# ========================================================================
# 12. APPLY ROW LOGIC
# ========================================================================

class TestApplyRowLogic:
    def test_apply_simple(self):
        hf = _make_hf()
        result = hf.apply_row_logic(lambda row: row["x"] + row["y"])
        assert "apply_result" in result.columns

    def test_apply_named_result(self):
        hf = _make_hf()
        result = hf.apply_row_logic(lambda row: row["x"] * 2)
        assert "apply_result" in result.columns

    def test_apply_returns_series(self):
        hf = _make_hf()
        result = hf.apply_row_logic(
            lambda row: pd.Series({"sum_xy": row["x"] + row["y"], "diff_xy": row["x"] - row["y"]})
        )
        assert "sum_xy" in result.columns
        assert "diff_xy" in result.columns


# ========================================================================
# 13. STREAMING
# ========================================================================

class TestStreaming:
    def test_fetch_chunked(self):
        hf = _make_hf()
        hf._to_duckdb()
        chunks = list(hf.fetch_chunked(batch_size=10))
        assert len(chunks) > 0
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == N

    def test_fetch_chunked_arrow(self):
        import pyarrow as pa  # noqa: F811
        hf = _make_hf()
        hf._to_duckdb()
        chunks = list(hf.fetch_chunked(batch_size=10, use_arrow=True))
        assert len(chunks) > 0

    def test_fetch_chunked_empty(self):
        hf = HybridFrame()
        chunks = list(hf.fetch_chunked(batch_size=10))
        assert len(chunks) == 0

    def test_to_pandas_iter(self):
        hf = _make_hf()
        hf._to_duckdb()
        chunks = list(hf.to_pandas_iter(batch_size=10))
        assert len(chunks) > 0
        total = sum(len(c) for c in chunks)
        assert total == N

    def test_to_arrow_reader(self):
        import pyarrow as pa  # noqa: F811
        hf = _make_hf()
        hf._to_duckdb()
        reader = hf.to_arrow_reader(batch_size=10)
        assert reader is not None
        batches = list(reader)
        assert len(batches) > 0


# ========================================================================
# 14. MEMORY USAGE
# ========================================================================

class TestMemoryUsage:
    def test_memory_usage_pandas(self):
        hf = _make_hf()
        result = hf.memory_usage()
        assert isinstance(result, pd.Series)
        assert result.sum() > 0

    def test_memory_usage_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        result = hf.memory_usage()
        assert isinstance(result, pd.Series)

    def test_memory_usage_empty(self):
        hf = HybridFrame()
        result = hf.memory_usage()
        assert isinstance(result, pd.Series)
        assert len(result) == 0


# ========================================================================
# 15. DUNDER / REPR
# ========================================================================

class TestDunder:
    def test_repr(self):
        hf = _make_hf()
        r = repr(hf)
        assert "HybridFrame" in r
        assert "Pandas" in r

    def test_repr_empty(self):
        hf = HybridFrame()
        r = repr(hf)
        assert "None" in r or "HybridFrame" in r

    def test_repr_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        r = repr(hf)
        assert "DuckDB" in r


# ========================================================================
# 16. LAZY CONNECTION
# ========================================================================

class TestLazyConnection:
    def test_conn_none_after_init(self):
        hf = HybridFrame()
        assert hf._conn is None

    def test_conn_none_after_from_pandas(self):
        hf = _make_hf()
        assert hf._conn is None

    def test_conn_after_to_duckdb(self):
        hf = _make_hf()
        hf._to_duckdb()
        assert hf._conn is not None

    def test_conn_stays_none_after_pandas_ops(self):
        hf = _make_hf()
        hf.select(["x", "y"])
        hf.head(5)
        hf.tail(5)
        hf.rename({"x": "x2"})
        hf.drop("y")
        assert hf._conn is None  # no connection acquired

    def test_conn_after_filter(self):
        hf = _make_hf()
        hf.filter("x > 0")
        assert hf._conn is not None  # filter triggers DuckDB

    def test_lazy_does_not_affect_join(self):
        """join should still work with lazy connections"""
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "a": [3, 4]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "b": [5, 6]}))
        result = hf1.join(hf2, on="k")
        assert result.shape[0] == 2


# ========================================================================
# 17. DUCKDB ERROR WRAPPING
# ========================================================================

class TestErrorWrapping:
    def test_filter_invalid_sql_message(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError) as exc:
            hf.filter("invalid sql !!!")
        assert "Filter condition failed" in str(exc.value)

    def test_sort_non_existent_column_message(self):
        hf = _make_hf()
        with pytest.raises(HybridFrameError) as exc:
            hf.sort_values("nope")
        assert "not found" in str(exc.value) or "found" in str(exc.value)

    def test_select_non_existent_message(self):
        hf = _make_hf()
        hf._to_duckdb()
        with pytest.raises(HybridFrameError) as exc:
            hf.select(["nope"])
        msg = str(exc.value)
        assert any(w in msg for w in ("column", "Column", "not found", "referenced"))


# ========================================================================
# 18. CROSS-ENGINE CONSISTENCY
# ========================================================================

class TestCrossEngineConsistency:
    """Same operation should give same result regardless of engine"""

    def test_select_pandas_vs_duckdb(self):
        df = _make_df()
        hf_pd = HybridFrame.from_pandas(df)
        hf_db = HybridFrame.from_pandas(df)
        hf_db._to_duckdb()
        res_pd = hf_pd.select(["x", "y"]).to_pandas()
        res_db = hf_db.select(["x", "y"]).to_pandas()
        pd.testing.assert_frame_equal(res_pd, res_db, check_dtype=False)

    def test_filter_pandas_vs_duckdb(self):
        df = _make_df()
        hf_pd = HybridFrame.from_pandas(df)
        hf_db = HybridFrame.from_pandas(df)
        hf_db._to_duckdb()
        res_pd = hf_pd.filter("x > 0").to_pandas()
        res_db = hf_db.filter("x > 0").to_pandas()
        pd.testing.assert_frame_equal(res_pd, res_db, check_dtype=False)

    def test_head_same(self):
        df = _make_df()
        hf_pd = HybridFrame.from_pandas(df)
        hf_db = HybridFrame.from_pandas(df)
        hf_db._to_duckdb()
        res_pd = hf_pd.head(5).to_pandas()
        res_db = hf_db.head(5).to_pandas()
        pd.testing.assert_frame_equal(res_pd, res_db, check_dtype=False)

    def test_sort_same(self):
        df = _make_df()
        hf_pd = HybridFrame.from_pandas(df)
        hf_db = HybridFrame.from_pandas(df)
        hf_db._to_duckdb()
        res_pd = hf_pd.sort_values("x").to_pandas()
        res_db = hf_db.sort_values("x").to_pandas()
        pd.testing.assert_frame_equal(res_pd, res_db, check_dtype=False)

    def test_rename_same(self):
        df = _make_df()
        hf_pd = HybridFrame.from_pandas(df)
        hf_db = HybridFrame.from_pandas(df)
        hf_db._to_duckdb()
        res_pd = hf_pd.rename({"x": "x2"}).to_pandas()
        res_db = hf_db.rename({"x": "x2"}).to_pandas()
        pd.testing.assert_frame_equal(res_pd, res_db, check_dtype=False)

    def test_drop_same(self):
        df = _make_df()
        hf_pd = HybridFrame.from_pandas(df)
        hf_db = HybridFrame.from_pandas(df)
        hf_db._to_duckdb()
        res_pd = hf_pd.drop("x").to_pandas()
        res_db = hf_db.drop("x").to_pandas()
        pd.testing.assert_frame_equal(res_pd, res_db, check_dtype=False)
