import gc
import time
import threading
import numpy as np
import pandas as pd
import duckdb
from hybrid_frame import HybridFrame, HybridFrameError

SMALL = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7.0, 8.0, 9.0]})
NANS = pd.DataFrame({"x": [1, None, 3], "y": [None, "b", None]})
CATS = pd.DataFrame({"cat": list("aabb"), "val": [1, 2, 3, 4]})
DUPS = pd.DataFrame({"id": [1, 1, 2, 2], "v": [10, 20, 30, 40]})

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

def test_basic_construction():
    hf = HybridFrame.from_pandas(SMALL)
    check("from_pandas engine=PANDAS", hf._engine.name == "PANDAS_DATAFRAME")
    check("from_pandas shape", hf.shape == (3, 3))
    check("from_pandas columns", list(hf.columns) == list("abc"))

    hd = HybridFrame.from_pandas(SMALL)._to_duckdb()
    check("_to_duckdb engine", hd._engine.name == "DUCKDB_RELATION")
    check("_to_duckdb shape", hd.shape == (3, 3))
    check("_to_duckdb columns", list(hd.columns) == list("abc"))

def test_filter_select_chain():
    hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
    hf.filter("a > 1")
    check("filter reduces size", hf.shape[0] == 2)
    hf.select(["a", "c"])
    check("select reduces columns", list(hf.columns) == ["a", "c"])

def test_sort_and_limit():
    hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
    hf.sort_values("a", ascending=True)
    check("sort_values engine=DuckDB", hf._engine.name == "DUCKDB_RELATION")
    arr = hf.to_pandas()["a"].values
    check("sort_values ascending", list(arr) == [1, 2, 3])
    hf2 = hf.sort_values("a", ascending=False).limit(2)
    check("limit 2", hf2.shape[0] == 2)
    vals = hf2.to_pandas()["a"].values
    check("limit+sort descending", list(vals) == [3, 2])

def test_head_tail():
    hf = HybridFrame.from_pandas(SMALL)._to_duckdb()
    h = hf.head(2)
    check("head returns correct rows", len(h) == 2)
    t = hf.tail(2)
    check("tail returns correct rows", len(t) == 2)

def test_groupby_agg():
    hf = HybridFrame.from_pandas(CATS)._to_duckdb()
    hf.groupby_agg(["cat"], {"val": "sum"})
    check("groupby_agg shape", hf.shape == (2, 2))
    check("groupby_agg has cat column", "cat" in hf.columns)
    check("groupby_agg has val_sum column", "val_sum" in hf.columns)
    pdf = hf.to_pandas().sort_values("cat").reset_index(drop=True)
    check("groupby_agg values", all(pdf["val_sum"] == [3, 7]))

def test_join_cross_connection():
    left = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "v": [10, 20]}))
    right = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 3], "w": [100, 300]}))
    left.join(right, on="k", how="inner")
    check("join cross-conn engine", left._engine.name == "DUCKDB_RELATION")
    check("join cross-conn shape", left.shape == (1, 4))
    check("join cross-conn has w", "w" in left.columns)

def test_join_same_connection():
    left = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 2], "v": [10, 20]}))
    left._to_duckdb()
    right = HybridFrame.from_pandas(pd.DataFrame({"k": [1, 3], "w": [100, 300]}))
    left.join(right, on="k", how="left")
    check("join same-conn shape", left.shape == (2, 4))

def test_rename():
    hf = HybridFrame.from_pandas(SMALL)
    hf.rename({"a": "aaa"})
    check("rename column present", "aaa" in hf.columns)
    check("rename old absent", "a" not in hf.columns)

def test_drop():
    hf = HybridFrame.from_pandas(SMALL)
    hf.drop(["c"])
    check("drop column absent", "c" not in hf.columns)
    check("drop column count", hf.shape[1] == 2)

def test_fillna():
    hf = HybridFrame.from_pandas(NANS)
    hf.fillna({"x": 0, "y": "missing"})
    pdf = hf.to_pandas()
    check("fillna x", pdf["x"].iloc[1] == 0)
    check("fillna y", pdf["y"].iloc[2] == "missing")

def test_isna_nunique():
    hf = HybridFrame.from_pandas(NANS)
    isna = hf.isna()
    check("isna detects nan", isna["x"].iloc[1])
    check("isna no false positive", not isna["x"].iloc[0])
    nun = hf.nunique()
    check("nunique returns Series", isinstance(nun, pd.Series))

def test_value_counts():
    hf = HybridFrame.from_pandas(pd.DataFrame({"x": ["a", "b", "a"]}))
    vc = hf.value_counts("x")
    check("value_counts returns Series", isinstance(vc, pd.Series))
    check("value_counts length", len(vc) == 2)

def test_write_csv_parquet(tmp_path):
    hf = HybridFrame.from_pandas(SMALL)
    p1 = tmp_path / "test_hf.csv"
    hf.write_csv(str(p1))
    check("write_csv file exists", p1.exists())
    reread = pd.read_csv(p1)
    check("write_csv roundtrip", reread.shape == (3, 3))

    p2 = tmp_path / "test_hf.parquet"
    hf.write_parquet(str(p2))
    check("write_parquet file exists", p2.exists())

def test_sql_method():
    hf = HybridFrame.from_pandas(SMALL)
    hf.sql("SELECT a + b AS s FROM self")
    check("sql adds computed column", "s" in hf.columns)
    check("sql correct value", hf.to_pandas()["s"].iloc[0] == 5)

def test_auto_transition():
    hf = HybridFrame.from_pandas(SMALL)
    hf._to_duckdb()
    before = hf._engine.name
    hf.to_pandas()
    after = hf._engine.name
    check("to_pandas transitions to Pandas", after == "PANDAS_DATAFRAME")
    hf.groupby_agg(["a"], {"b": "sum"})
    check("agg auto-transitions to DuckDB", hf._engine.name == "DUCKDB_RELATION")

def test_duckdb_pandas_consistency():
    pdf = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    hf1 = HybridFrame.from_pandas(pdf)
    hf2 = HybridFrame.from_pandas(pdf)
    hf2._to_duckdb()
    hf1.filter("x > 2")
    hf2.filter("x > 2")
    pdf1 = hf1.to_pandas()
    pdf2 = hf2.to_pandas().reset_index(drop=True)
    check("pandas vs duckdb filter same", pdf1["x"].tolist() == pdf2["x"].tolist())

def test_thread_safety():
    HybridFrame.close_all_connections()
    df = pd.DataFrame({"a": np.arange(100), "b": np.random.rand(100)})
    hf = HybridFrame.from_pandas(df)
    errors = []
    lock = threading.Lock()

    def worker(ident):
        try:
            for _ in range(20):
                local = HybridFrame.from_pandas(df)
                local._to_duckdb()
                local.filter("a > 50")
                local.sort_values("b", ascending=False)
                local.head(5)
                local.to_pandas()
        except Exception as e:
            with lock:
                errors.append((ident, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    check("thread safety: no errors", len(errors) == 0)
    still_alive = sum(1 for t in threads if t.is_alive())
    check("thread safety: all threads finished", still_alive == 0)

def test_errors_propagated():
    hf = HybridFrame.from_pandas(SMALL)
    try:
        hf.groupby_agg(["nonexistent"], {"a": "sum"})
        check("error on bad group column", False)
    except HybridFrameError:
        check("error on bad group column", True)
    except Exception:
        check("error on bad group column (wrong type)", False)

def test_empty_filter():
    df = pd.DataFrame({"x": [1, 2, 3]})
    hf = HybridFrame.from_pandas(df)
    hf.filter("x > 10")
    check("empty filter returns 0 rows", hf.shape[0] == 0)

def test_from_pandas_copy_independence():
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    hf = HybridFrame.from_pandas(pdf)
    pdf["a"] = [99, 99, 99]
    check("copy independence", hf.to_pandas()["a"].iloc[0] == 1)

def test_multi_column_groupby():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [10, 20, 30, 40]})
    hf = HybridFrame.from_pandas(df)
    hf.groupby_agg(["a", "b"], {"c": "sum"})
    check("multi-column groupby shape", hf.shape == (4, 3))
    check("multi-column has a", "a" in hf.columns)
    check("multi-column has b", "b" in hf.columns)
    check("multi-column has c_sum", "c_sum" in hf.columns)

def test_connection_cleanup():
    import psutil
    import os
    proc = psutil.Process(os.getpid())
    before = proc.num_fds()
    frames = [HybridFrame.from_pandas(pd.DataFrame({"x": range(100)})) for _ in range(20)]
    for f in frames:
        if f._conn is not None:
            f._conn.close()
    gc.collect()
    after = proc.num_fds()
    check("connection cleanup (fd count stable)", after <= before + 5)

def test_to_pandas_consistency():
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    hf = HybridFrame.from_pandas(pdf)
    hf._to_duckdb()
    result = hf.to_pandas()
    check("to_pandas roundtrip same rows", len(result) == len(pdf))
    for col in pdf.columns:
        check(f"to_pandas roundtrip col {col}", list(result[col]) == list(pdf[col]))

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    tmp = tempfile.mkdtemp()
    print(f"Temp dir: {tmp}")

    tests = [
        ("basic_construction", test_basic_construction),
        ("filter_select_chain", test_filter_select_chain),
        ("sort_and_limit", test_sort_and_limit),
        ("head_tail", test_head_tail),
        ("groupby_agg", test_groupby_agg),
        ("join_cross_connection", test_join_cross_connection),
        ("join_same_connection", test_join_same_connection),
        ("rename", test_rename),
        ("drop", test_drop),
        ("fillna", test_fillna),
        ("isna_nunique", test_isna_nunique),
        ("value_counts", test_value_counts),
        ("write_csv_parquet", lambda: test_write_csv_parquet(Path(tmp))),
        ("sql_method", test_sql_method),
        ("auto_transition", test_auto_transition),
        ("duckdb_pandas_consistency", test_duckdb_pandas_consistency),
        ("thread_safety", test_thread_safety),
        ("errors_propagated", test_errors_propagated),
        ("empty_filter", test_empty_filter),
        ("from_pandas_copy_independence", test_from_pandas_copy_independence),
        ("multi_column_groupby", test_multi_column_groupby),
        ("connection_cleanup", test_connection_cleanup),
        ("to_pandas_consistency", test_to_pandas_consistency),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name} (exception: {e})")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed:
        import sys
        sys.exit(1)
