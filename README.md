# HybridFrame

A high-performance dual-engine DataFrame that transparently shifts between DuckDB (lazy/out-of-core) and Pandas (materialised/in-memory).

## Installation

```bash
pip install hybrid-frame
```

With optional dependencies:

```bash
pip install hybrid-frame[arrow]       # zero-copy Arrow materialisation
pip install hybrid-frame[memory-guard] # psutil-based OOM protection
pip install hybrid-frame[ml]           # scikit-learn for ML export
pip install hybrid-frame[all]          # everything
```

## Quick Start

```python
from hybrid_frame import HybridFrame
import pandas as pd

# Wrap an existing DataFrame
hf = HybridFrame.from_pandas(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))

# Lazy DuckDB operations — zero Pandas copies
hf.filter("x > 1").sort_values("y", ascending=False)

# Chain into Pandas for ML feature engineering
result = hf.assign(x2=lambda df: df["x"] * 2).to_pandas()

# Stream a CSV lazily from disk
hf = HybridFrame.from_csv("large_file.csv")
print(hf.filter("age > 21").shape)
```

## Features

- **Zero-copy engine transitions** — stays in Pandas mode for simple operations (head/tail/select/rename/drop), auto-transitions to DuckDB for complex SQL operations (filter/sort/groupby/join)
- **Lazy DuckDB connections** — no connection overhead until DuckDB is actually needed
- **Out-of-core streaming** — `fetch_chunked()`, `to_pandas_iter()`, `to_arrow_reader()` for memory-efficient processing
- **ML-ready export** — `to_ml_ready(target_column)` returns `(X, y)` tuple
- **File I/O** — `from_csv()`, `from_parquet()`, `write_csv()`, `write_parquet()`
- **Set operations** — `union`, `intersect`, `except_`
- **Cumulative operations** — `diff`, `cumsum`, `cumprod`, `cummin`, `cummax`
- **Value imputation** — `fillna`, `dropna`, `replace`, `clip`, `time_series_impute`
- **DuckDB SQL passthrough** — `sql("SELECT ... FROM self WHERE ...")`
- **Memory guard** — automatic OOM protection for large materialisations

## Documentation

Full API reference is available in the class docstrings. Run `help(HybridFrame)` or visit [GitHub](https://github.com/shell-bay/hybrid-frame).
