"""Property-based tests for HybridFrame using Hypothesis."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from hybrid_frame import HybridFrame

# ---------------------------------------------------------------------------
# Strategy: small DataFrames with mixed columns
# ---------------------------------------------------------------------------

@st.composite
def dataframes(draw):
    n_rows = draw(st.integers(min_value=1, max_value=20))
    n_cols = draw(st.integers(min_value=1, max_value=5))
    columns = draw(
        st.lists(
            st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                    min_size=1, max_size=8),
            min_size=n_cols, max_size=n_cols, unique=True,
        )
    )
    data = {}
    for col in columns:
        dtype = draw(st.sampled_from(["int", "float", "string", "bool"]))
        if dtype == "int":
            data[col] = draw(st.lists(
                st.one_of(st.integers(min_value=-100, max_value=100), st.just(None)),
                min_size=n_rows, max_size=n_rows,
            ))
        elif dtype == "float":
            data[col] = draw(st.lists(
                st.one_of(
                    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False,
                              allow_infinity=False),
                    st.just(None),
                ),
                min_size=n_rows, max_size=n_rows,
            ))
        elif dtype == "string":
            data[col] = draw(st.lists(
                st.one_of(st.text(alphabet="abc", max_size=5), st.just(None)),
                min_size=n_rows, max_size=n_rows,
            ))
        else:  # bool
            data[col] = draw(st.lists(
                st.one_of(st.booleans(), st.just(None)),
                min_size=n_rows, max_size=n_rows,
            ))
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestHypothesisProperties:

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_filter_partition(self, df):
        hf = HybridFrame.from_pandas(df)
        numeric_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64")]
        if not numeric_cols or hf.shape[0] == 0:
            return
        col = numeric_cols[0]
        hf_gt = hf.filter(f'"{col}" > 0')
        hf_le = hf.filter(f'NOT ("{col}" > 0) OR "{col}" IS NULL')
        total = hf_gt.shape[0] + hf_le.shape[0]
        assert total == hf.shape[0], (
            f"filter partition: {hf_gt.shape[0]} + {hf_le.shape[0]} != {hf.shape[0]}"
        )

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_fillna_eliminates_nans(self, df):
        hf = HybridFrame.from_pandas(df)
        hf.fillna(0)
        assert hf.isna().sum().sum() == 0, "fillna(0) should eliminate all NaN"

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_dropna_eliminates_nans(self, df):
        hf = HybridFrame.from_pandas(df)
        hf.dropna(how="any")
        assert hf.isna().sum().sum() == 0, "dropna(how='any') should leave no NaN"

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_select_all_columns(self, df):
        hf = HybridFrame.from_pandas(df)
        selected = hf.select(hf.columns)
        assert selected.columns == hf.columns, "select(all columns) should preserve columns"

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_rename_empty(self, df):
        hf = HybridFrame.from_pandas(df)
        renamed = hf.rename({})
        assert renamed.columns == hf.columns, "rename({}) should not change columns"

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_drop_empty(self, df):
        hf = HybridFrame.from_pandas(df)
        dropped = hf.drop([])
        assert dropped.shape == hf.shape, "drop([]) should not change shape"

    @settings(max_examples=200)
    @given(df=dataframes())
    def test_roundtrip_duckdb_pandas(self, df):
        hf = HybridFrame.from_pandas(df)
        hf._to_duckdb()
        result = hf.to_pandas().reset_index(drop=True)
        original = df.reset_index(drop=True)
        for col in df.columns:
            if df[col].dtype in ("int64", "float64"):
                res_vals = result[col].tolist()
                orig_vals = original[col].tolist()
                for r, o in zip(res_vals, orig_vals):
                    if pd.isna(r) and pd.isna(o):
                        continue
                    assert r == o, (
                        f"Roundtrip mismatch in column {col}: {r} != {o}"
                    )
            elif df[col].dtype == "bool":
                assert result[col].tolist() == original[col].tolist(), (
                    f"Roundtrip mismatch in bool column {col}"
                )
