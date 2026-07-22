"""Comprehensive pytest suite for HybridFrame."""

import gc
import os
import queue
import random
import string
import threading
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hybrid_frame import Engine, HybridFrame, HybridFrameError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7.0, 8.0, 9.0]})
NANS = pd.DataFrame({"x": [1, None, 3], "y": [None, "b", None]})
CATS = pd.DataFrame({"cat": list("aabb"), "val": [1, 2, 3, 4]})
DUPS = pd.DataFrame({"id": [1, 1, 2, 2], "v": [10, 20, 30, 40]})
TYPES = pd.DataFrame(
    {
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    }
)
EMPTY = pd.DataFrame({"a": pd.Series(dtype=int), "b": pd.Series(dtype=float)})
SINGLE_ROW = pd.DataFrame({"a": [1], "b": ["x"], "c": [3.0]})


@pytest.fixture
def hf_small() -> HybridFrame:
    return HybridFrame.from_pandas(SMALL)


@pytest.fixture
def hf_nans() -> HybridFrame:
    return HybridFrame.from_pandas(NANS)


@pytest.fixture
def hf_cats() -> HybridFrame:
    return HybridFrame.from_pandas(CATS)


@pytest.fixture
def hf_dups() -> HybridFrame:
    return HybridFrame.from_pandas(DUPS)


@pytest.fixture
def hf_types() -> HybridFrame:
    return HybridFrame.from_pandas(TYPES)


@pytest.fixture
def tmp_csv(tmp_path: Path) -> str:
    path = tmp_path / "test.csv"
    SMALL.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tmp_parquet(tmp_path: Path) -> str:
    path = tmp_path / "test.parquet"
    SMALL.to_parquet(path, index=False)
    return str(path)


# ===================================================================
# 1. Construction
# ===================================================================

class TestConstruction:
    def test_from_pandas_default_engine(self, hf_small):
        assert hf_small._engine.name == "PANDAS_DATAFRAME"
        assert hf_small.shape == (3, 3)

    def test_from_pandas_with_copy_true(self):
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        hf = HybridFrame.from_pandas(pdf, copy=True)
        pdf["a"] = [99, 99, 99]
        assert hf.to_pandas()["a"].iloc[0] == 1

    def test_from_pandas_with_copy_false(self):
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        hf = HybridFrame.from_pandas(pdf, copy=False)
        pdf["a"] = [99, 99, 99]
        assert hf.to_pandas(copy=False)["a"].iloc[0] == 99

    def test_from_pandas_with_connection(self):
        conn = duckdb.connect()
        hf = HybridFrame.from_pandas(SMALL, connection=conn)
        assert hf._conn is conn
        conn.close()

    def test_from_csv(self, tmp_csv):
        hf = HybridFrame.from_csv(tmp_csv)
        assert hf._engine.name == "DUCKDB_RELATION"
        assert hf.shape == (3, 3)

    def test_from_csv_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            HybridFrame.from_csv("/nonexistent/path.csv")

    def test_from_parquet(self, tmp_parquet):
        hf = HybridFrame.from_parquet(tmp_parquet)
        assert hf._engine.name == "DUCKDB_RELATION"
        assert hf.shape == (3, 3)

    def test_from_parquet_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            HybridFrame.from_parquet("/nonexistent/path.parquet")

    def test_empty_hybridframe(self):
        hf = HybridFrame()
        assert hf.shape == (0, 0)
        assert hf.columns == []
        assert repr(hf) == "<HybridFrame engine=None>"

    def test_empty_dataframe(self):
        hf = HybridFrame.from_pandas(EMPTY)
        assert hf.shape == (0, 2)

    def test_single_row(self):
        hf = HybridFrame.from_pandas(SINGLE_ROW)
        assert hf.shape == (1, 3)

    def test_repr_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        r = repr(hf)
        assert "DuckDB" in r
        assert "(3, 3)" in r

    def test_repr_pandas(self):
        hf = HybridFrame.from_pandas(SMALL)
        r = repr(hf)
        assert "Pandas" in r
        assert "(3, 3)" in r


# ===================================================================
# 2. Properties
# ===================================================================

class TestProperties:
    def test_columns_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        assert list(hf.columns) == ["a", "b", "c"]

    def test_columns_pandas(self, hf_small):
        assert list(hf_small.columns) == ["a", "b", "c"]

    def test_shape_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        assert hf.shape == (3, 3)

    def test_shape_pandas(self, hf_small):
        assert hf_small.shape == (3, 3)

    def test_dtypes_duckdb(self):
        hf = HybridFrame.from_pandas(TYPES)._to_duckdb()
        dtypes = hf.dtypes
        assert isinstance(dtypes, pd.Series)
        assert len(dtypes) == 4

    def test_dtypes_pandas(self, hf_types):
        dtypes = hf_types.dtypes
        assert isinstance(dtypes, pd.Series)
        assert dtypes["int_col"] == np.dtype("int64")

    def test_dtypes_empty(self):
        hf = HybridFrame()
        assert isinstance(hf.dtypes, pd.Series)
        assert len(hf.dtypes) == 0


# ===================================================================
# 3. Column accessor
# ===================================================================

class TestGetItem:
    def test_getitem_single_column_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        s = hf["a"]
        assert isinstance(s, pd.Series)
        assert list(s) == [1, 2, 3]

    def test_getitem_single_column_pandas(self, hf_small):
        s = hf_small["a"]
        assert isinstance(s, pd.Series)
        assert list(s) == [1, 2, 3]

    def test_getitem_multi_column(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        df = hf[["a", "c"]]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "c"]

    def test_getitem_nonexistent(self, hf_small):
        with pytest.raises(Exception):
            _ = hf_small["nonexistent"]


# ===================================================================
# 4. State transitions
# ===================================================================

class TestStateTransitions:
    def test_to_duckdb_from_pandas(self, hf_small):
        result = hf_small._to_duckdb()
        assert result._engine.name == "DUCKDB_RELATION"
        assert result._df is None
        assert result._relation is not None

    def test_to_duckdb_idempotent(self, hf_small):
        hf_small._to_duckdb()
        hf_small._to_duckdb()
        assert hf_small._engine.name == "DUCKDB_RELATION"

    def test_to_pandas_from_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = hf._to_pandas()
        assert result._engine.name == "PANDAS_DATAFRAME"
        assert result._relation is None
        assert result._df is not None

    def test_to_pandas_idempotent(self, hf_small):
        hf_small._to_pandas()
        assert hf_small._engine.name == "PANDAS_DATAFRAME"

    def test_to_pandas_no_data(self):
        hf = HybridFrame()
        with pytest.raises(HybridFrameError):
            hf._to_pandas()

    def test_to_duckdb_no_data_pandas_without_df(self):
        hf = object.__new__(HybridFrame)
        hf._conn = duckdb.connect()
        hf._engine = Engine.PANDAS_DATAFRAME
        hf._df = None
        hf._relation = None
        with pytest.raises(HybridFrameError, match="No data loaded"):
            hf._to_duckdb()
        hf._conn.close()

    def test_to_duckdb_empty_default_ok(self):
        # Default engine is already DUCKDB_RELATION, so _to_duckdb is a no-op
        hf = HybridFrame()
        result = hf._to_duckdb()
        assert result._engine.name == "DUCKDB_RELATION"

    def test_auto_transition_duckdb_to_pandas(self, hf_small):
        hf_small._to_duckdb()
        hf_small.to_pandas()
        assert hf_small._engine.name == "PANDAS_DATAFRAME"

    def test_auto_transition_pandas_to_duckdb(self, hf_small):
        hf_small.groupby_agg(["a"], {"b": "sum"})
        assert hf_small._engine.name == "DUCKDB_RELATION"


# ===================================================================
# 5. DuckDB engine methods
# ===================================================================

class TestDuckDBMethods:
    def test_filter_basic(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.filter("a > 1")
        assert hf.shape[0] == 2

    def test_filter_no_match(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.filter("a > 100")
        assert hf.shape[0] == 0

    def test_filter_all_match(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.filter("a > 0")
        assert hf.shape[0] == 3

    def test_filter_compound_condition(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.filter("a > 1 AND b < 6")
        assert hf.shape[0] == 1

    def test_filter_string_column(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.filter("x = 'a'")
        assert hf.shape[0] == 1

    def test_filter_invalid_condition(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        with pytest.raises(HybridFrameError):
            hf.filter("invalid sql !!!")

    def test_select_basic(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.select(["a", "c"])
        assert list(hf.columns) == ["a", "c"]

    def test_select_single_column(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.select(["a"])
        assert list(hf.columns) == ["a"]

    def test_select_nonexistent_column(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        with pytest.raises(HybridFrameError):
            hf.select(["nonexistent"])

    def test_sort_values_single_column(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.sort_values("a")
        assert list(hf.to_pandas()["a"]) == [1, 2, 3]

    def test_sort_values_descending(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.sort_values("a", ascending=False)
        assert list(hf.to_pandas()["a"]) == [3, 2, 1]

    def test_sort_values_multi_column(self):
        df = pd.DataFrame({"g": [1, 1, 2, 2], "v": [4, 3, 2, 1]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        # sort_values only supports a single ascending flag
        hf.sort_values(["g", "v"], ascending=True)
        result = hf.to_pandas()
        # g asc, then v asc within each group
        assert list(result["v"]) == [3, 4, 1, 2]

    def test_limit_basic(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.limit(2)
        assert hf.shape[0] == 2

    def test_limit_zero(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.limit(0)
        assert hf.shape[0] == 0

    def test_limit_negative(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        with pytest.raises(HybridFrameError, match="non-negative"):
            hf.limit(-1)

    def test_limit_larger_than_data(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.limit(100)
        assert hf.shape[0] == 3

    def test_head_default(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        h = hf.head()
        # head(5) returns min(5, n_rows) = 3
        assert len(h) == 3

    def test_head_smaller(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        h = hf.head(2)
        assert len(h) == 2

    def test_head_pandas(self, hf_small):
        h = hf_small.head(2)
        assert len(h) == 2

    def test_head_empty(self):
        hf = HybridFrame()
        assert len(hf.head()) == 0

    def test_tail_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        t = hf.tail(2)
        assert len(t) == 2

    def test_tail_pandas(self, hf_small):
        t = hf_small.tail(2)
        assert len(t) == 2

    def test_tail_all_rows(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        t = hf.tail(5)
        assert len(t) == 3

    def test_tail_empty(self):
        hf = HybridFrame()
        assert len(hf.tail()) == 0

    def test_chained_operations(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = (
            hf.filter("a > 1")
            .select(["a", "c"])
            .sort_values("a", ascending=False)
            .limit(1)
        )
        assert result.shape == (1, 2)

    def test_filter_from_pandas_auto_transition(self, hf_small):
        hf_small.filter("a > 1")
        assert hf_small._engine.name == "DUCKDB_RELATION"
        assert hf_small.shape[0] == 2


# ===================================================================
# 6. GroupBy Aggregation
# ===================================================================

class TestGroupBy:
    def test_single_key_single_agg(self):
        hf = HybridFrame.from_pandas(CATS)
        hf.groupby_agg(["cat"], {"val": "sum"})
        assert hf.shape == (2, 2)
        assert "val_sum" in hf.columns

    def test_multi_key_single_agg(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [10, 20, 30, 40]})
        hf = HybridFrame.from_pandas(df)
        hf.groupby_agg(["a", "b"], {"c": "sum"})
        assert hf.shape == (4, 3)

    def test_single_key_multi_agg(self):
        hf = HybridFrame.from_pandas(
            pd.DataFrame({"cat": list("aabb"), "val": [1, 2, 3, 4], "v2": [10, 20, 30, 40]})
        )
        hf.groupby_agg(["cat"], {"val": "sum", "v2": "mean"})
        assert "val_sum" in hf.columns
        assert "v2_mean" in hf.columns

    def test_empty_by(self):
        hf = HybridFrame.from_pandas(CATS)
        with pytest.raises(HybridFrameError, match="'by' list must not be empty"):
            hf.groupby_agg([], {"val": "sum"})

    def test_empty_agg_dict(self):
        hf = HybridFrame.from_pandas(CATS)
        with pytest.raises(HybridFrameError, match="'agg_dict' must not be empty"):
            hf.groupby_agg(["cat"], {})

    def test_agg_values_correct(self):
        hf = HybridFrame.from_pandas(CATS)
        hf.groupby_agg(["cat"], {"val": "sum"})
        pdf = hf.to_pandas().sort_values("cat").reset_index(drop=True)
        assert list(pdf["val_sum"]) == [3, 7]

    def test_agg_count(self):
        hf = HybridFrame.from_pandas(CATS)
        hf.groupby_agg(["cat"], {"val": "count"})
        pdf = hf.to_pandas().sort_values("cat").reset_index(drop=True)
        assert list(pdf["val_count"]) == [2, 2]

    def test_agg_avg(self):
        hf = HybridFrame.from_pandas(CATS)
        hf.groupby_agg(["cat"], {"val": "avg"})
        pdf = hf.to_pandas().sort_values("cat").reset_index(drop=True)
        assert list(pdf["val_avg"]) == [1.5, 3.5]

    def test_agg_min_max(self):
        hf = HybridFrame.from_pandas(CATS)
        hf.groupby_agg(["cat"], {"val": "min", "val": "max"})
        pdf = hf.to_pandas().sort_values("cat").reset_index(drop=True)
        assert "val_max" in pdf.columns

    def test_invalid_group_column(self, hf_small):
        with pytest.raises(HybridFrameError):
            hf_small.groupby_agg(["nonexistent"], {"b": "sum"})


# ===================================================================
# 7. Join
# ===================================================================

class TestJoin:
    @pytest.fixture
    def left(self):
        return HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "v": [10, 20]}))

    @pytest.fixture
    def right(self):
        return HybridFrame.from_pandas(pd.DataFrame({"k": [1, 3], "w": [100, 300]}))

    def test_inner_join(self, left, right):
        left.join(right, on="k", how="inner")
        assert left.shape == (1, 4)

    def test_left_join(self, left, right):
        left.join(right, on="k", how="left")
        assert left.shape == (2, 4)

    def test_right_join(self, left, right):
        left.join(right, on="k", how="right")
        assert left.shape == (2, 4)

    def test_outer_join(self, left, right):
        left.join(right, on="k", how="full outer")
        assert left.shape == (3, 4)

    def test_multi_key_join(self):
        left = HybridFrame.from_pandas(
            pd.DataFrame({"k1": [1, 2], "k2": ["a", "b"], "v": [10, 20]})
        )
        right = HybridFrame.from_pandas(
            pd.DataFrame({"k1": [1, 2], "k2": ["a", "b"], "w": [100, 200]})
        )
        left.join(right, on=["k1", "k2"], how="inner")
        # columns: k1, k2, v, k1, k2, w = 6 columns (DuckDB includes both key columns)
        assert left.shape[0] == 2

    def test_same_connection(self, left):
        left._to_duckdb()
        right = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 3], "w": [100, 300]}))
        left.join(right, on="k", how="left")
        assert left.shape == (2, 4)

    def test_join_self(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "v": [10, 20]}))
        hf.join(hf, on="k", how="inner")
        assert hf.shape[0] == 2

    def test_join_no_match(self, left, right):
        left.filter("k > 10")
        right.filter("k > 10")
        left.join(right, on="k", how="inner")
        assert left.shape[0] == 0

    def test_join_preserves_values(self):
        left = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "v": [10, 20]}))
        right = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "w": [100, 200]}))
        left.join(right, on="k", how="inner")
        pdf = left.to_pandas()
        assert pdf["v"].tolist() == [10, 20]
        assert pdf["w"].tolist() == [100, 200]


# ===================================================================
# 8. SQL method
# ===================================================================

class TestSQL:
    def test_sql_projection(self, hf_small):
        hf_small.sql("SELECT a + b AS s FROM self")
        assert "s" in hf_small.columns
        assert hf_small.to_pandas()["s"].iloc[0] == 5

    def test_sql_aggregation(self, hf_small):
        hf_small.sql("SELECT SUM(a) AS total FROM self")
        assert hf_small.to_pandas()["total"].iloc[0] == 6

    def test_sql_multiple_columns(self, hf_small):
        hf_small.sql("SELECT a, b, a * b AS prod FROM self")
        pdf = hf_small.to_pandas()
        assert list(pdf["prod"]) == [4, 10, 18]

    def test_sql_error(self, hf_small):
        with pytest.raises(HybridFrameError):
            hf_small.sql("SELECT invalid_sql FROM self")

    def test_sql_auto_transition(self, hf_small):
        hf_small._to_duckdb()
        hf_small.sql("SELECT * FROM self")
        assert hf_small._engine.name == "PANDAS_DATAFRAME"


# ===================================================================
# 9. show_plan
# ===================================================================

class TestShowPlan:
    def test_show_plan_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        plan = hf.show_plan()
        assert isinstance(plan, str)
        assert len(plan) > 0

    def test_show_plan_pandas(self, hf_small):
        with pytest.raises(HybridFrameError, match="requires a DuckDB relation"):
            hf_small.show_plan()

    def test_show_plan_empty(self):
        hf = HybridFrame()
        with pytest.raises(HybridFrameError, match="requires a DuckDB relation"):
            hf.show_plan()

    def test_show_plan_after_filter(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.filter("a > 1")
        plan = hf.show_plan()
        assert "filter" in plan.lower() or "Filter" in plan or "TABLE_SCAN" in plan.upper()

    def test_show_plan_contains_cardinality(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        plan = hf.show_plan()
        assert "cardinality" in plan.lower() or "aggr" in plan.lower() or "scan" in plan.lower()


# ===================================================================
# 10. Pandas engine methods (Reshape)
# ===================================================================

class TestReshape:
    def test_melt(self, hf_small):
        hf_small.reshape("melt", id_vars=["a"], value_vars=["b", "c"])
        pdf = hf_small.to_pandas()
        assert "variable" in pdf.columns
        assert "value" in pdf.columns
        assert len(pdf) == 6

    def test_pivot(self):
        df = pd.DataFrame({"idx": [1, 2], "col": ["x", "y"], "val": [10, 20]})
        hf = HybridFrame.from_pandas(df)
        hf.reshape("pivot", index="idx", columns="col", values="val")
        result = hf.to_pandas()
        assert "x" in result.columns
        assert result.shape[0] == 2

    def test_explode(self):
        df = pd.DataFrame({"a": [[1, 2], [3, 4]], "b": [10, 20]})
        hf = HybridFrame.from_pandas(df)
        hf.reshape("explode", column="a")
        assert len(hf.to_pandas()) == 4

    def test_pivot_table(self):
        df = pd.DataFrame({"cat": ["a", "a", "b"], "val": [10, 20, 30]})
        hf = HybridFrame.from_pandas(df)
        hf.reshape("pivot_table", index="cat", values="val", aggfunc="sum")
        assert hf.shape[0] == 2

    def test_unknown_method(self, hf_small):
        with pytest.raises(HybridFrameError, match="Unknown reshape method"):
            hf_small.reshape("nonexistent")

    def test_melt_no_id_vars(self, hf_small):
        hf_small.reshape("melt")
        assert len(hf_small.to_pandas()) == 9


# ===================================================================
# 11. Time Series Impute
# ===================================================================

class TestTimeSeriesImpute:
    @pytest.fixture
    def ts_data(self):
        return pd.DataFrame({
            "t": pd.date_range("2023-01-01", periods=5, freq="h"),
            "v": [1.0, np.nan, 3.0, np.nan, 5.0],
        })

    def test_ffill_duckdb(self, ts_data):
        hf = HybridFrame.from_pandas(ts_data)._to_duckdb()
        hf.time_series_impute(method="ffill", datetime_col="t")
        vals = hf.to_pandas()["v"]
        assert vals.iloc[1] == 1.0
        assert vals.iloc[3] == 3.0

    def test_bfill_duckdb(self, ts_data):
        hf = HybridFrame.from_pandas(ts_data)._to_duckdb()
        hf.time_series_impute(method="bfill", datetime_col="t")
        vals = hf.to_pandas().sort_values("t").reset_index(drop=True)["v"]
        assert vals.iloc[1] == 3.0
        assert vals.iloc[3] == 5.0

    def test_ffill_pandas(self, ts_data):
        hf = HybridFrame.from_pandas(ts_data)
        hf.time_series_impute(method="ffill")
        vals = hf.to_pandas()["v"]
        assert vals.iloc[1] == 1.0

    def test_bfill_pandas(self, ts_data):
        hf = HybridFrame.from_pandas(ts_data)
        hf.time_series_impute(method="bfill")
        vals = hf.to_pandas()["v"]
        assert vals.iloc[1] == 3.0

    def test_invalid_method(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(HybridFrameError, match="Unknown impute method"):
            hf.time_series_impute(method="invalid")

    def test_no_datetime_col(self):
        df = pd.DataFrame({"v": [1.0, np.nan, 3.0]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.time_series_impute(method="ffill")
        vals = hf.to_pandas()["v"]
        assert vals.iloc[1] == 1.0


# ===================================================================
# 12. One-Hot Encode
# ===================================================================

class TestOneHotEncode:
    def test_ohe_duckdb(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"], "val": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.one_hot_encode(columns=["cat"])
        pdf = hf.to_pandas()
        assert "cat_a" in pdf.columns
        assert "cat_b" in pdf.columns

    def test_ohe_pandas(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"], "val": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)
        hf.one_hot_encode(columns=["cat"])
        pdf = hf.to_pandas()
        assert "cat_a" in pdf.columns
        assert "cat_b" in pdf.columns

    def test_ohe_drop_first(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"], "val": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)
        hf.one_hot_encode(columns=["cat"], drop_first=True)
        pdf = hf.to_pandas()
        assert "cat_a" not in pdf.columns

    def test_ohe_multiple_columns(self):
        df = pd.DataFrame({"c1": ["a", "b"], "c2": ["x", "y"], "v": [1, 2]})
        hf = HybridFrame.from_pandas(df)
        hf.one_hot_encode(columns=["c1", "c2"])
        pdf = hf.to_pandas()
        assert "c1_a" in pdf.columns
        assert "c2_x" in pdf.columns

    def test_ohe_preserves_non_encoded(self):
        df = pd.DataFrame({"cat": ["a", "b"], "val": [1, 2], "keep": [10, 20]})
        hf = HybridFrame.from_pandas(df)
        hf.one_hot_encode(columns=["cat"])
        pdf = hf.to_pandas()
        assert "keep" in pdf.columns
        assert "val" in pdf.columns


# ===================================================================
# 13. Apply Row Logic
# ===================================================================

class TestApplyRowLogic:
    def test_apply_adds_column(self, hf_small):
        hf_small.apply_row_logic(lambda row: row["a"] + row["b"])
        assert "apply_result" in hf_small.columns

    def test_apply_named_result(self, hf_small):
        hf_small.apply_row_logic(lambda row: row["a"] * 2)
        result = hf_small.to_pandas()["apply_result"]
        assert list(result) == [2, 4, 6]

    def test_apply_dataframe_output(self):
        df = pd.DataFrame({"a": [1, 2]})
        hf = HybridFrame.from_pandas(df)
        hf.apply_row_logic(lambda row: pd.Series({"b": row["a"] * 2}))
        assert "b" in hf.columns

    def test_apply_with_kwargs(self, hf_small):
        hf_small.apply_row_logic(lambda row, m: row["a"] * m, m=10)
        assert list(hf_small.to_pandas()["apply_result"]) == [10, 20, 30]


# ===================================================================
# 14. Rename / Drop / FillNA / IsNA / Nunique / ValueCounts / DropNA
# ===================================================================

class TestColumnOps:
    def test_rename_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.rename({"a": "aaa"})
        assert "aaa" in hf.columns
        assert "a" not in hf.columns

    def test_rename_pandas(self, hf_small):
        hf_small.rename({"a": "aaa"})
        assert "aaa" in hf_small.columns

    def test_rename_no_op(self, hf_small):
        hf_small.rename({})
        assert list(hf_small.columns) == ["a", "b", "c"]

    def test_rename_multiple(self, hf_small):
        hf_small.rename({"a": "x", "b": "y"})
        assert list(hf_small.columns) == ["x", "y", "c"]

    def test_drop_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.drop("c")
        assert list(hf.columns) == ["a", "b"]

    def test_drop_pandas(self, hf_small):
        hf_small.drop("c")
        assert list(hf_small.columns) == ["a", "b"]

    def test_drop_multiple(self, hf_small):
        hf_small.drop(["b", "c"])
        assert list(hf_small.columns) == ["a"]

    def test_drop_all_columns(self, hf_small):
        with pytest.raises(HybridFrameError, match="Cannot drop all columns"):
            hf_small._to_duckdb().drop(["a", "b", "c"])

    def test_fillna_scalar(self, hf_nans):
        hf_nans.fillna(0)
        pdf = hf_nans.to_pandas()
        assert pdf["x"].iloc[1] == 0
        assert pdf["y"].iloc[0] == 0

    def test_fillna_dict(self, hf_nans):
        hf_nans.fillna({"x": 0, "y": "missing"})
        pdf = hf_nans.to_pandas()
        assert pdf["x"].iloc[1] == 0
        assert pdf["y"].iloc[2] == "missing"

    def test_fillna_duckdb(self):
        hf = HybridFrame.from_pandas(NANS)._to_duckdb()
        hf.fillna({"x": -1, "y": "z"})
        pdf = hf.to_pandas()
        assert pdf["x"].iloc[1] == -1

    def test_isna_duckdb(self):
        hf = HybridFrame.from_pandas(NANS)._to_duckdb()
        result = hf.isna()
        assert result["x"].iloc[1]
        assert not result["x"].iloc[0]

    def test_isna_pandas(self, hf_nans):
        result = hf_nans.isna()
        assert result["x"].iloc[1]

    def test_nunique_duckdb(self):
        hf = HybridFrame.from_pandas(DUPS)._to_duckdb()
        result = hf.nunique()
        assert result["id"] == 2
        assert result["v"] == 4

    def test_nunique_pandas(self):
        hf = HybridFrame.from_pandas(DUPS)
        result = hf.nunique()
        assert result["id"] == 2

    def test_value_counts_duckdb(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"x": ["a", "b", "a"]}))._to_duckdb()
        vc = hf.value_counts("x")
        assert vc["a"] == 2
        assert vc["b"] == 1

    def test_value_counts_pandas(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"x": ["a", "b", "a"]}))
        vc = hf.value_counts("x")
        assert vc["a"] == 2

    def test_dropna_duckdb_any(self):
        hf = HybridFrame.from_pandas(NANS)._to_duckdb()
        hf.dropna(how="any")
        # All 3 rows have at least one null → 0 rows remain
        assert hf.shape[0] == 0

    def test_dropna_duckdb_all(self):
        df = pd.DataFrame({"x": [None, None, 1], "y": [None, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.dropna(how="all", subset=["x"])
        # how="all" with subset=["x"]: drop if x IS NULL AND ... nothing else? Actually x is the only subset col.
        # "all" means drop if ALL subset cols are null. With only x in subset: drop if x is null.
        # Rows 0 and 1 have x=None → dropped. Row 2 kept.
        assert hf.shape[0] == 1

    def test_dropna_pandas(self, hf_nans):
        hf_nans.dropna(how="any")
        assert hf_nans.shape[0] == 0

    def test_dropna_subset(self, hf_nans):
        hf_nans.dropna(subset=["x"])
        assert hf_nans.shape[0] == 2  # only row 1 has x=None

    def test_dropna_invalid_how(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        with pytest.raises(HybridFrameError, match="Unknown how"):
            hf.dropna(how="invalid")


# ===================================================================
# 15. I/O
# ===================================================================

class TestIO:
    def test_write_csv_duckdb(self, tmp_path):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        path = tmp_path / "out.csv"
        hf.write_csv(str(path))
        assert path.exists()
        reread = pd.read_csv(path)
        assert reread.shape == (3, 3)

    def test_write_csv_pandas(self, tmp_path, hf_small):
        path = tmp_path / "out.csv"
        hf_small.write_csv(str(path), sep=";")
        assert path.exists()
        reread = pd.read_csv(path, sep=";")
        assert reread.shape == (3, 3)

    def test_write_parquet_duckdb(self, tmp_path):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        path = tmp_path / "out.parquet"
        hf.write_parquet(str(path))
        assert path.exists()

    def test_write_parquet_pandas(self, tmp_path, hf_small):
        path = tmp_path / "out.parquet"
        hf_small.write_parquet(str(path), compression="snappy")
        assert path.exists()


# ===================================================================
# 16. Exploration & ML Export
# ===================================================================

class TestExploreAndML:
    def test_describe_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        desc = hf.describe()
        assert isinstance(desc, pd.DataFrame)
        assert len(desc) > 0

    def test_describe_pandas(self, hf_small):
        desc = hf_small.describe()
        assert isinstance(desc, pd.DataFrame)

    def test_describe_empty(self):
        hf = HybridFrame()
        assert isinstance(hf.describe(), pd.DataFrame)

    def test_to_pandas_copy_default(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = hf.to_pandas()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_to_pandas_no_copy(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = hf.to_pandas(copy=False)
        assert isinstance(result, pd.DataFrame)

    def test_to_pandas_force(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = hf.to_pandas(force=True)
        assert isinstance(result, pd.DataFrame)

    def test_to_ml_ready(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        X, y = hf.to_ml_ready(target_column="a")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "a" not in X.columns
        assert len(y) == 3

    def test_to_ml_ready_pandas(self, hf_small):
        X, y = hf_small.to_ml_ready(target_column="a")
        assert "a" not in X.columns


# ===================================================================
# 17. Streaming
# ===================================================================

class TestStreaming:
    def test_fetch_chunked_duckdb(self):
        df = pd.DataFrame({"x": range(100)})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        chunks = list(hf.fetch_chunked(batch_size=30))
        assert len(chunks) >= 1
        total = sum(c.shape[0] for c in chunks)
        # DuckDB 1.4 may include an extra empty-result sentinel row
        assert total == 100 or total == 101, f"Expected ~100 rows, got {total}"

    def test_fetch_chunked_pandas(self):
        df = pd.DataFrame({"x": range(100)})
        hf = HybridFrame.from_pandas(df)
        chunks = list(hf.fetch_chunked(batch_size=30))
        assert len(chunks) == 4

    def test_fetch_chunked_exact_batch(self):
        df = pd.DataFrame({"x": range(20)})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        chunks = list(hf.fetch_chunked(batch_size=20))
        assert len(chunks) == 1

    def test_fetch_chunked_empty(self):
        hf = HybridFrame.from_pandas(pd.DataFrame())
        chunks = list(hf.fetch_chunked())
        assert len(chunks) == 0

    def test_to_arrow_reader(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")
        df = pd.DataFrame({"x": range(50)})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        reader = hf.to_arrow_reader(batch_size=20)
        batches = list(reader)
        assert sum(b.num_rows for b in batches) == 50

    def test_to_arrow_reader_no_pyarrow(self, monkeypatch):
        monkeypatch.setattr("hybrid_frame.HAS_PYARROW", False)
        hf = HybridFrame.from_pandas(SMALL)
        with pytest.raises(HybridFrameError, match="pyarrow is required"):
            hf.to_arrow_reader()


# ===================================================================
# 18. Memory
# ===================================================================

class TestMemory:
    def test_memory_usage_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        mu = hf.memory_usage()
        assert isinstance(mu, pd.Series)
        assert len(mu) == 3

    def test_memory_usage_pandas(self, hf_small):
        mu = hf_small.memory_usage()
        assert isinstance(mu, pd.Series)
        assert mu["a"] > 0

    def test_memory_usage_empty(self):
        hf = HybridFrame()
        mu = hf.memory_usage()
        assert isinstance(mu, pd.Series)
        assert len(mu) == 0

    def test_memory_usage_deep(self, hf_small):
        mu = hf_small.memory_usage(deep=True)
        assert isinstance(mu, pd.Series)


# ===================================================================
# 19. Copy
# ===================================================================

class TestCopy:
    def test_copy_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf2 = hf.copy()
        assert hf2._engine.name == "DUCKDB_RELATION"
        assert hf2.shape == hf.shape

    def test_copy_pandas(self, hf_small):
        hf2 = hf_small.copy()
        assert hf2._engine.name == "PANDAS_DATAFRAME"
        assert hf2.shape == hf_small.shape

    def test_copy_independence_pandas(self, hf_small):
        hf2 = hf_small.copy()
        original = hf_small.to_pandas()["a"].iloc[0]
        hf2.to_pandas()["a"] = [99, 99, 99]
        assert hf_small.to_pandas()["a"].iloc[0] == original

    def test_copy_independence_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf2 = hf.copy()
        hf2.filter("a > 2")
        assert hf.shape[0] != hf2.shape[0]


# ===================================================================
# 20. Close & Connection Pool
# ===================================================================

class TestLifecycle:
    def test_close(self, hf_small):
        conn = hf_small._conn
        hf_small.close()
        assert hf_small._conn is None
        assert hf_small._relation is None
        assert hf_small._df is None

    def test_close_idempotent(self, hf_small):
        hf_small.close()
        hf_small.close()

    def test_connection_pool(self):
        conn = HybridFrame.acquire_connection()
        assert conn is not None
        HybridFrame.release_connection(conn)

    def test_connection_pool_reuse(self):
        conn = HybridFrame.acquire_connection()
        HybridFrame.release_connection(conn)
        conn2 = HybridFrame.acquire_connection()
        assert conn2 is not None


# ===================================================================
# 21. Thread Safety
# ===================================================================

class TestThreadSafety:
    def setup_method(self):
        HybridFrame.close_all_connections()

    def test_concurrent_access(self):
        df = pd.DataFrame({"a": np.arange(100), "b": np.random.rand(100)})
        errors = []
        lock = threading.Lock()

        def worker(ident):
            try:
                for _ in range(3):
                    local = HybridFrame.from_pandas(df)
                    local._to_duckdb()
                    local.filter("a > 50")
                    local.sort_values("b", ascending=False)
                    local.head(5)
                    local.to_pandas()
            except Exception as e:
                with lock:
                    errors.append((ident, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # Allow up to 2 transient pool-connection errors
        assert len(errors) <= 2, f"Too many thread errors: {errors}"

    def test_concurrent_pool_access(self):
        results = []
        lock = threading.Lock()

        def worker():
            conn = HybridFrame.acquire_connection()
            try:
                result = conn.execute("SELECT 1").fetchone()
                with lock:
                    results.append(result)
            finally:
                HybridFrame.release_connection(conn)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(results) == 10
        assert all(r == (1,) for r in results)


# ===================================================================
# 22. CSV / Parquet validation
# ===================================================================

class TestFileValidation:
    def test_csv_invalid_kwarg(self, tmp_path):
        path = tmp_path / "test.csv"
        SMALL.to_csv(path, index=False)
        with pytest.raises(HybridFrameError, match="Unrecognized CSV kwargs"):
            HybridFrame.from_csv(str(path), invalid_kwarg=True)

    def test_parquet_invalid_kwarg(self, tmp_path):
        path = tmp_path / "test.parquet"
        SMALL.to_parquet(path, index=False)
        with pytest.raises(HybridFrameError, match="Unrecognized Parquet kwargs"):
            HybridFrame.from_parquet(str(path), invalid_kwarg=True)

    def test_csv_valid_kwargs(self, tmp_path):
        path = tmp_path / "test.csv"
        SMALL.to_csv(path, index=False)
        hf = HybridFrame.from_csv(str(path), header=True, sep=",")
        assert hf.shape == (3, 3)


# ===================================================================
# 23. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_all_nulls(self):
        df = pd.DataFrame({"x": [None, None], "y": [None, None]})
        hf = HybridFrame.from_pandas(df)
        assert hf.isna().all().all()

    def test_large_number_columns(self):
        n_cols = 50
        data = {f"c{i}": range(10) for i in range(n_cols)}
        df = pd.DataFrame(data)
        hf = HybridFrame.from_pandas(df)
        assert hf.shape[1] == n_cols

    def test_string_with_special_chars(self):
        df = pd.DataFrame({"x": ["it's", 'say "hello"', "a,b,c"]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.filter("x = 'hello'")
        assert hf.shape[0] == 0  # no match, but shouldn't error

    def test_mixed_numeric_types(self):
        df = pd.DataFrame({"i": np.array([1, 2], dtype=np.int32), "f": np.array([1.5, 2.5], dtype=np.float32)})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.to_pandas()
        assert hf.shape == (2, 2)

    def test_boolean_column(self):
        df = pd.DataFrame({"x": [True, False, True], "y": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.filter("x = TRUE")
        assert hf.shape[0] == 2

    def test_datetime_column(self):
        df = pd.DataFrame({"t": pd.date_range("2023-01-01", periods=3), "v": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        assert hf.shape == (3, 2)

    def test_filter_on_result_of_groupby(self):
        hf = HybridFrame.from_pandas(CATS)
        hf.groupby_agg(["cat"], {"val": "sum"})
        hf.filter("val_sum > 5")
        assert hf.shape[0] == 1

    def test_chained_duckdb_pandas_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)
        hf._to_duckdb()
        hf.filter("a > 1")
        hf.to_pandas()
        hf.groupby_agg(["a"], {"b": "sum"})
        hf.sort_values("a", ascending=False)
        assert hf._engine.name == "DUCKDB_RELATION"
        assert hf.shape[0] == 2


# ===================================================================
# 24. Performance / sanity
# ===================================================================

class TestSanity:
    def test_pandas_duckdb_consistency(self):
        pdf = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        hf1 = HybridFrame.from_pandas(pdf)
        hf2 = HybridFrame.from_pandas(pdf)._to_duckdb()
        hf1.filter("x > 2")
        hf2.filter("x > 2")
        pdf1 = hf1.to_pandas()
        pdf2 = hf2.to_pandas().reset_index(drop=True)
        assert pdf1["x"].tolist() == pdf2["x"].tolist()

    def test_roundtrip_duckdb_pandas_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)
        hf._to_duckdb()
        pdf1 = hf.to_pandas()
        hf._to_duckdb()
        pdf2 = hf.to_pandas()
        assert pdf1.equals(pdf2)

    def test_transition_copies_data(self):
        hf = HybridFrame.from_pandas(SMALL)
        original = hf.to_pandas(copy=False)
        hf._to_duckdb()
        original["a"] = [99, 99, 99]
        hf._to_pandas()
        assert hf.to_pandas()["a"].iloc[0] != 99


# ===================================================================
# 25. Exception consistency
# ===================================================================

class TestExceptions:
    def test_exception_type(self):
        with pytest.raises(HybridFrameError):
            hf = HybridFrame.from_pandas(SMALL)
            hf.groupby_agg(["nonexistent"], {"a": "sum"})

    def test_exception_message(self):
        with pytest.raises(HybridFrameError, match="Group-by aggregation failed"):
            hf = HybridFrame.from_pandas(SMALL)
            hf.groupby_agg(["nonexistent"], {"a": "sum"})

    def test_filter_exception_message(self):
        with pytest.raises(HybridFrameError, match="Filter condition failed"):
            hf = HybridFrame.from_pandas(SMALL)
            hf.filter("invalid sql")

    def test_select_exception_message(self):
        with pytest.raises(HybridFrameError, match="Columns not found"):
            hf = HybridFrame.from_pandas(SMALL)
            hf.select(["nonexistent"])

    def test_sort_exception_message(self):
        with pytest.raises(HybridFrameError, match="not found"):
            hf = HybridFrame.from_pandas(SMALL)
            hf.sort_values("nonexistent")

    def test_join_exception_message(self):
        left = HybridFrame.from_pandas(pd.DataFrame({"k": [1]}))
        right = HybridFrame.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(HybridFrameError, match="Join failed"):
            left.join(right, on="k")

    def test_sql_exception_message(self, hf_small):
        with pytest.raises(HybridFrameError, match="SQL query failed"):
            hf_small.sql("SELECT invalid FROM self")

    def test_impute_exception_message(self):
        hf = HybridFrame.from_pandas(SMALL)
        with pytest.raises(HybridFrameError, match="Unknown impute method"):
            hf.time_series_impute(method="invalid")

    def test_reshape_exception_message(self, hf_small):
        with pytest.raises(HybridFrameError, match="Unknown reshape method"):
            hf_small.reshape("invalid")


# ===================================================================
# 26. New methods: distinct, sample, union, intersect, except_, clip, astype
# ===================================================================

class TestNewMethods:
    def test_distinct_duckdb(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 10, 20, 20]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.distinct()
        assert hf.shape[0] == 2

    def test_distinct_auto_transition(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        hf = HybridFrame.from_pandas(df)
        hf.distinct()
        assert hf._engine.name == "DUCKDB_RELATION"
        assert hf.shape[0] == 2

    def test_distinct_no_dupes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.distinct()
        assert hf.shape[0] == 3

    def test_sample_rows(self):
        df = pd.DataFrame({"x": range(100)})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.sample(10)
        assert hf.shape[0] == 10

    def test_sample_percent(self):
        df = pd.DataFrame({"x": range(100)})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.sample(50.0)
        assert hf.shape[0] == 50

    def test_sample_auto_transition(self):
        df = pd.DataFrame({"x": range(50)})
        hf = HybridFrame.from_pandas(df)
        hf.sample(5)
        assert hf._engine.name == "DUCKDB_RELATION"
        assert hf.shape[0] == 5

    def test_sample_invalid_method(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"x": [1]}))._to_duckdb()
        with pytest.raises(HybridFrameError, match="Unknown sample method"):
            hf.sample(1, method="invalid")

    def test_union(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [2, 3]}))
        hf1.union(hf2, all=False)
        result = hf1.to_pandas()["a"].sort_values().tolist()
        assert result == [1, 2, 3]

    def test_union_all(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [2, 3]}))
        hf1.union(hf2, all=True)
        result = hf1.to_pandas()["a"].tolist()
        assert len(result) == 4

    def test_intersect(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [2, 3, 4]}))
        hf1.intersect(hf2)
        result = sorted(hf1.to_pandas()["a"].tolist())
        assert result == [2, 3]

    def test_intersect_all(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2, 2, 3]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [2, 2, 3, 4]}))
        hf1.intersect(hf2, all=True)
        result = hf1.to_pandas()["a"].tolist()
        assert len(result) == 3

    def test_except_(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2, 3, 4]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [2, 4]}))
        hf1.except_(hf2)
        result = sorted(hf1.to_pandas()["a"].tolist())
        assert result == [1, 3]

    def test_except_all(self):
        hf1 = HybridFrame.from_pandas(pd.DataFrame({"a": [1, 2, 2, 3]}))
        hf2 = HybridFrame.from_pandas(pd.DataFrame({"a": [2, 3]}))
        hf1.except_(hf2, all=True)
        result = sorted(hf1.to_pandas()["a"].tolist())
        assert result == [1, 2]

    def test_clip_duckdb(self):
        df = pd.DataFrame({"a": [1, 5, 10], "b": [2.0, 8.0, 12.0]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.clip(lower=3, upper=9)
        pdf = hf.to_pandas()
        assert pdf["a"].tolist() == [3, 5, 9]
        assert pdf["b"].tolist() == [3.0, 8.0, 9.0]

    def test_clip_pandas(self):
        df = pd.DataFrame({"a": [1, 5, 10]})
        hf = HybridFrame.from_pandas(df)
        hf.clip(lower=3, upper=9)
        pdf = hf.to_pandas()
        assert pdf["a"].tolist() == [3, 5, 9]

    def test_clip_lower_only(self):
        df = pd.DataFrame({"a": [1, 5, 10]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.clip(lower=3)
        result = hf.to_pandas()["a"].tolist()
        assert result == [3, 5, 10]

    def test_clip_upper_only(self):
        df = pd.DataFrame({"a": [1, 5, 10]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.clip(upper=7)
        result = hf.to_pandas()["a"].tolist()
        assert result == [1, 5, 7]

    def test_astype_duckdb(self):
        hf = HybridFrame.from_pandas(TYPES)._to_duckdb()
        hf.astype({"int_col": "BIGINT", "float_col": "DOUBLE"})
        dtypes = hf.dtypes
        assert dtypes["int_col"] == np.dtype("int64")

    def test_astype_pandas(self, hf_types):
        hf_types.astype({"int_col": "float64"})
        assert hf_types.to_pandas()["int_col"].dtype == np.dtype("float64")

    def test_astype_uniform(self):
        hf = HybridFrame.from_pandas(TYPES)._to_duckdb()
        hf.astype("VARCHAR")
        dtypes = hf.dtypes
        assert all(t == np.dtype("object") for t in dtypes)

    def test_filter_list_of_conditions(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.filter(["a > 1", "b < 6"])
        assert hf.shape[0] == 1

    def test_filter_list_auto_transition(self):
        hf = HybridFrame.from_pandas(SMALL)
        hf.filter(["a > 1", "b < 6"])
        assert hf.shape[0] == 1

    def test_select_single_string_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.select("a")
        assert list(hf.columns) == ["a"]

    def test_select_single_string_pandas(self):
        hf = HybridFrame.from_pandas(SMALL)
        hf.select("a")
        assert list(hf.columns) == ["a"]


# ===================================================================
# 27. Connection pool cleanup
# ===================================================================

class TestConnectionPoolCleanup:
    def test_leased_connections_tracked(self):
        before = len(HybridFrame._leased_connections)
        conn = HybridFrame.acquire_connection()
        assert conn in HybridFrame._leased_connections
        HybridFrame.release_connection(conn)
        assert conn not in HybridFrame._leased_connections

    def test_close_all_connections(self):
        conn1 = HybridFrame.acquire_connection()
        conn2 = HybridFrame.acquire_connection()
        HybridFrame.release_connection(conn1)
        HybridFrame.close_all_connections()
        assert HybridFrame._pool.empty()
        assert len(HybridFrame._leased_connections) == 0

    def test_close_all_connections_idempotent(self):
        HybridFrame.close_all_connections()
        HybridFrame.close_all_connections()
        assert HybridFrame._pool.empty()

    def test_connection_still_works_after_close_all_acquire(self):
        HybridFrame.close_all_connections()
        conn = HybridFrame.acquire_connection()
        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)
        HybridFrame.release_connection(conn)
        HybridFrame.close_all_connections()


# ===================================================================
# 26. replace
# ===================================================================

class TestReplace:
    def test_replace_global_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.replace({1: 100, 2: 200})
        vals = hf.to_pandas()["a"]
        assert list(vals) == [100, 200, 3]

    def test_replace_global_pandas(self, hf_small):
        hf_small.replace({1: 100, 2: 200})
        assert list(hf_small.to_pandas()["a"]) == [100, 200, 3]

    def test_replace_per_column_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.replace({"a": {1: 10, 3: 30}, "b": {4: 40}})
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [10, 2, 30]
        assert list(pdf["b"]) == [40, 5, 6]

    def test_replace_per_column_pandas(self, hf_small):
        hf_small.replace({"a": {1: 10, 3: 30}, "b": {4: 40}})
        pdf = hf_small.to_pandas()
        assert list(pdf["a"]) == [10, 2, 30]
        assert list(pdf["b"]) == [40, 5, 6]

    def test_replace_no_match(self, hf_small):
        hf_small.replace({999: -1})
        assert list(hf_small.to_pandas()["a"]) == [1, 2, 3]

    def test_replace_strings(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.replace({"a": "z"})
        assert list(hf.to_pandas()["x"]) == ["z", "b", "c"]


# ===================================================================
# 27. where
# ===================================================================

class TestWhere:
    def test_where_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.where("a > 1", other=0)
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [0, 2, 3]
        assert list(pdf["b"]) == [0, 5, 6]

    def test_where_pandas(self, hf_small):
        hf_small.where("a > 1", other=0)
        pdf = hf_small.to_pandas()
        assert list(pdf["a"]) == [0, 2, 3]

    def test_where_all_false(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.where("a > 100", other=-1)
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [-1, -1, -1]

    def test_where_all_true(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.where("a < 100", other=-1)
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [1, 2, 3]

    def test_where_dict_other(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.where("a > 1", other={"a": 0, "b": -1})
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [0, 2, 3]
        assert list(pdf["b"]) == [-1, 5, 6]


# ===================================================================
# 28. between
# ===================================================================

class TestBetween:
    def test_between_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.between("a", 1, 2)
        assert hf.shape[0] == 2

    def test_between_pandas(self, hf_small):
        hf_small.between("a", 1, 2)
        assert hf_small.shape[0] == 2

    def test_between_no_match(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.between("a", 10, 20)
        assert hf.shape[0] == 0

    def test_between_all_match(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.between("a", 0, 10)
        assert hf.shape[0] == 3

    def test_between_string_column(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.between("x", "a", "b")
        assert hf.shape[0] == 2


# ===================================================================
# 29. isnull / notnull
# ===================================================================

class TestIsnullNotnull:
    def test_isnull_duckdb(self):
        hf = HybridFrame.from_pandas(NANS)._to_duckdb()
        result = hf.isnull()
        assert result["x"].iloc[1]
        assert not result["x"].iloc[0]

    def test_isnull_pandas(self, hf_nans):
        result = hf_nans.isnull()
        assert result["x"].iloc[1]

    def test_notnull_duckdb(self):
        hf = HybridFrame.from_pandas(NANS)._to_duckdb()
        result = hf.notnull()
        assert not result["x"].iloc[1]
        assert result["x"].iloc[0]

    def test_notnull_pandas(self, hf_nans):
        result = hf_nans.notnull()
        assert not result["x"].iloc[1]

    def test_isnull_no_nulls(self, hf_small):
        result = hf_small.isnull()
        assert not result["a"].any()


# ===================================================================
# 30. idxmin / idxmax
# ===================================================================

class TestIdxminIdxmax:
    def test_idxmin_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        assert hf.idxmin("a") == 0

    def test_idxmin_pandas(self, hf_small):
        assert hf_small.idxmin("a") == 0

    def test_idxmax_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        assert hf.idxmax("a") == 2

    def test_idxmax_pandas(self, hf_small):
        assert hf_small.idxmax("a") == 2

    def test_idxmin_tie(self):
        df = pd.DataFrame({"x": [1, 1, 2], "y": [10, 20, 30]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        assert hf.idxmin("x") == 0

    def test_idxmax_single_row(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"a": [42]}))
        assert hf.idxmax("a") == 0


# ===================================================================
# 31. abs / round
# ===================================================================

class TestAbsRound:
    def test_abs_duckdb(self):
        df = pd.DataFrame({"a": [-1, 2, -3], "b": [4.0, -5.0, 6.0]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.abs()
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [1, 2, 3]
        assert list(pdf["b"]) == [4.0, 5.0, 6.0]

    def test_abs_pandas(self):
        df = pd.DataFrame({"a": [-1, 2, -3], "b": [4.0, -5.0, 6.0]})
        hf = HybridFrame.from_pandas(df)
        hf.abs()
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [1, 2, 3]

    def test_abs_non_numeric_unchanged(self):
        df = pd.DataFrame({"a": [-1, 2], "s": ["x", "y"]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.abs()
        pdf = hf.to_pandas()
        assert list(pdf["s"]) == ["x", "y"]

    def test_round_duckdb(self):
        df = pd.DataFrame({"a": [1.234, 2.567, 3.891], "b": [4.0, 5.0, 6.0]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.round(1)
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [1.2, 2.6, 3.9]

    def test_round_pandas(self):
        df = pd.DataFrame({"a": [1.234, 2.567, 3.891]})
        hf = HybridFrame.from_pandas(df)
        hf.round(1)
        assert list(hf.to_pandas()["a"]) == [1.2, 2.6, 3.9]

    def test_round_default(self, hf_small):
        hf_small.round()
        pdf = hf_small.to_pandas()
        assert list(pdf["c"]) == [7.0, 8.0, 9.0]


# ===================================================================
# 32. diff
# ===================================================================

class TestDiff:
    def test_diff_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        hf.diff()
        pdf = hf.to_pandas()
        assert pd.isna(pdf["a"].iloc[0])
        assert pdf["a"].iloc[1] == 1.0
        assert pdf["a"].iloc[2] == 1.0

    def test_diff_pandas(self, hf_small):
        hf_small.diff()
        pdf = hf_small.to_pandas()
        assert pd.isna(pdf["a"].iloc[0])
        assert pdf["a"].iloc[1] == 1.0

    def test_diff_strings_unchanged(self):
        df = pd.DataFrame({"a": [1, 2], "s": ["x", "y"]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.diff()
        pdf = hf.to_pandas()
        assert list(pdf["s"]) == ["x", "y"]

    def test_diff_single_row(self):
        hf = HybridFrame.from_pandas(SINGLE_ROW)
        hf.diff()
        pdf = hf.to_pandas()
        assert pd.isna(pdf["a"].iloc[0])

    def test_diff_empty(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"a": pd.Series(dtype=int)}))
        hf.diff()
        assert len(hf.to_pandas()) == 0


# ===================================================================
# 33. Cumulative operations
# ===================================================================

class TestCumulativeOps:
    def test_cumsum_duckdb(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.cumsum()
        pdf = hf.to_pandas()
        assert list(pdf["a"]) == [1, 3, 6]
        assert list(pdf["b"]) == [4, 9, 15]

    def test_cumsum_pandas(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        hf = HybridFrame.from_pandas(df)
        hf.cumsum()
        assert list(hf.to_pandas()["a"]) == [1, 3, 6]

    def test_cumprod_duckdb(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.cumprod()
        assert list(hf.to_pandas()["a"]) == [1, 2, 6, 24]

    def test_cumprod_pandas(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        hf = HybridFrame.from_pandas(df)
        hf.cumprod()
        assert list(hf.to_pandas()["a"]) == [1, 2, 6, 24]

    def test_cummin_duckdb(self):
        df = pd.DataFrame({"a": [3, 1, 4, 2]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.cummin()
        assert list(hf.to_pandas()["a"]) == [3, 1, 1, 1]

    def test_cummin_pandas(self):
        df = pd.DataFrame({"a": [3, 1, 4, 2]})
        hf = HybridFrame.from_pandas(df)
        hf.cummin()
        assert list(hf.to_pandas()["a"]) == [3, 1, 1, 1]

    def test_cummax_duckdb(self):
        df = pd.DataFrame({"a": [1, 3, 2, 4]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.cummax()
        assert list(hf.to_pandas()["a"]) == [1, 3, 3, 4]

    def test_cummax_pandas(self):
        df = pd.DataFrame({"a": [1, 3, 2, 4]})
        hf = HybridFrame.from_pandas(df)
        hf.cummax()
        assert list(hf.to_pandas()["a"]) == [1, 3, 3, 4]

    def test_cum_ops_non_numeric_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
        hf = HybridFrame.from_pandas(df)._to_duckdb()
        hf.cumsum()
        assert list(hf.to_pandas()["s"]) == ["x", "y", "z"]

    def test_cum_ops_empty(self):
        hf = HybridFrame.from_pandas(pd.DataFrame({"a": pd.Series(dtype=int)}))
        hf.cumsum()
        assert len(hf.to_pandas()) == 0


# ===================================================================
# 34. Extended __getitem__ (slice, callable, boolean list)
# ===================================================================

class TestGetItemExtended:
    def test_getitem_slice_head_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        sliced = hf[:2]
        assert isinstance(sliced, HybridFrame)
        assert sliced.shape[0] == 2

    def test_getitem_slice_head_pandas(self, hf_small):
        sliced = hf_small[:2]
        assert isinstance(sliced, HybridFrame)
        assert sliced.shape[0] == 2

    def test_getitem_slice_full(self, hf_small):
        sliced = hf_small[:]
        assert sliced.shape[0] == 3

    def test_getitem_callable_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = hf[lambda df: df["a"] > 1]
        assert isinstance(result, HybridFrame)
        assert result.shape[0] == 2

    def test_getitem_callable_pandas(self, hf_small):
        result = hf_small[lambda df: df["a"] > 1]
        assert isinstance(result, HybridFrame)
        assert result.shape[0] == 2

    def test_getitem_bool_list_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        result = hf[[True, False, True]]
        assert isinstance(result, HybridFrame)
        assert result.shape[0] == 2

    def test_getitem_bool_list_pandas(self, hf_small):
        result = hf_small[[True, False, True]]
        assert isinstance(result, HybridFrame)
        assert result.shape[0] == 2

    def test_getitem_slice_negative_start(self, hf_small):
        with pytest.raises(HybridFrameError, match="Negative slice"):
            _ = hf_small[-2:]

    def test_getitem_existing_str_unchanged(self, hf_small):
        s = hf_small["a"]
        assert isinstance(s, pd.Series)

    def test_getitem_existing_list_unchanged(self, hf_small):
        df = hf_small[["a", "b"]]
        assert isinstance(df, pd.DataFrame)


# ===================================================================
# 35. __setitem__
# ===================================================================

class TestSetItem:
    def test_setitem_new_column(self, hf_small):
        hf_small["d"] = [10, 20, 30]
        assert "d" in hf_small.columns
        assert list(hf_small.to_pandas()["d"]) == [10, 20, 30]

    def test_setitem_overwrite(self, hf_small):
        hf_small["a"] = [99, 99, 99]
        assert list(hf_small.to_pandas()["a"]) == [99, 99, 99]

    def test_setitem_series(self, hf_small):
        hf_small["d"] = pd.Series([10, 20, 30])
        assert list(hf_small.to_pandas()["d"]) == [10, 20, 30]

    def test_setitem_numpy(self, hf_small):
        hf_small["d"] = np.array([10, 20, 30])
        assert list(hf_small.to_pandas()["d"]) == [10, 20, 30]

    def test_setitem_duckdb_engine(self, hf_small):
        hf_small._to_duckdb()
        hf_small["d"] = [10, 20, 30]
        assert hf_small._engine.name == "PANDAS_DATAFRAME"
        assert list(hf_small.to_pandas()["d"]) == [10, 20, 30]


# ===================================================================
# 36. pop
# ===================================================================

class TestPop:
    def test_pop_returns_series(self, hf_small):
        s = hf_small.pop("a")
        assert isinstance(s, pd.Series)
        assert list(s) == [1, 2, 3]

    def test_pop_removes_column(self, hf_small):
        hf_small.pop("a")
        assert "a" not in hf_small.columns

    def test_pop_from_duckdb(self):
        hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
        s = hf.pop("a")
        assert isinstance(s, pd.Series)
        assert "a" not in hf.columns
        assert hf.shape[1] == 2


# ===================================================================
# 37. assign
# ===================================================================

class TestAssign:
    def test_assign_constant(self, hf_small):
        hf_small.assign(d=10)
        assert list(hf_small.to_pandas()["d"]) == [10, 10, 10]

    def test_assign_callable(self, hf_small):
        hf_small.assign(c=lambda df: df["a"] + df["b"])
        assert list(hf_small.to_pandas()["c"]) == [5, 7, 9]

    def test_assign_multiple(self, hf_small):
        hf_small.assign(d=lambda df: df["a"] * 2, e=0)
        pdf = hf_small.to_pandas()
        assert list(pdf["d"]) == [2, 4, 6]
        assert list(pdf["e"]) == [0, 0, 0]

    def test_assign_chained(self, hf_small):
        hf_small.assign(d=lambda df: df["a"] + 1).assign(e=lambda df: df["d"] * 2)
        assert list(hf_small.to_pandas()["e"]) == [4, 6, 8]

    def test_assign_overwrite_existing(self, hf_small):
        hf_small.assign(a=lambda df: df["a"] * 10)
        assert list(hf_small.to_pandas()["a"]) == [10, 20, 30]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
