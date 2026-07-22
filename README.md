# HybridFrame

<p align="center">
  <img src="https://img.shields.io/pypi/v/hybrid-frame.svg" alt="PyPI version">
  <img src="https://img.shields.io/pypi/pyversions/hybrid-frame.svg" alt="Python versions">
  <img src="https://img.shields.io/pypi/l/hybrid-frame.svg" alt="License">
</p>

<p align="center">
  <strong>The fastest way to work with data in Python</strong>
</p>

<p align="center">
  HybridFrame unifies DuckDB (out-of-core, lazy) and Pandas (in-memory, ML) into a single DataFrame API. 
  Zero-copy transitions. Zero configuration. Just fast data.
</p>

---

## Quick Start

```python
from hybrid_frame import HybridFrame
import pandas as pd

# Create from Pandas
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "salary": [50000, 60000, 70000, 55000, 65000],
    "dept": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"]
})

hf = HybridFrame.from_pandas(df)

# Lazy DuckDB operations — zero copies
result = (
    hf
    .filter("age > 28")
    .sort_values("salary", ascending=False)
    .select(["name", "salary", "dept"])
)

# Materialize to Pandas for ML
X, y = result.to_ml_ready("salary")
```

## Installation

```bash
pip install hybrid-frame
```

With optional dependencies:

```bash
pip install hybrid-frame[arrow]          # Zero-copy Arrow materialization
pip install hybrid-frame[memory-guard]   # OOM protection with psutil
pip install hybrid-frame[ml]             # scikit-learn for ML export
pip install hybrid-frame[all]            # Everything
```

## Demo Video

https://github.com/user-attachments/assets/hybridframe-demo.mp4

*(Watch the 37-second overview of HybridFrame architecture and performance)*

## Why HybridFrame?

| Operation | Pandas | HybridFrame | Speedup |
|-----------|--------|-------------|---------|
| Filter 1M rows | 19ms | 12ms | **1.5x** |
| Sort + Head 5 | 78ms | 3ms | **26x** |
| GroupBy Sum | 13ms | 17ms | 0.7x |
| Head 5 | 0.01ms | 0.08ms | 0.1x |

*Benchmark on 1M rows, warm start, full round-trip including `.to_pandas()`*

**Where HybridFrame wins:**
- Complex analytical queries (filter, sort, groupby, join)
- Chained operations (filter → sort → head)
- Large datasets that don't fit in memory

**Where Pandas wins:**
- Simple metadata operations (head, tail, select, rename)
- ML feature engineering (assign, one-hot encode, apply)

## Features

### Core Operations

```python
# Filter rows
hf.filter("age > 25")
hf.filter(["age > 25", "salary > 60000"])

# Select columns
hf.select(["name", "age"])

# Sort
hf.sort_values("salary", ascending=False)
hf.sort_values(["dept", "salary"], ascending=[True, False])

# Limit rows
hf.head(10)
hf.tail(10)
hf.limit(100)
```

### Aggregations

```python
# Single aggregation
hf.groupby_agg(["dept"], {"salary": "sum"})

# Multiple aggregations
hf.groupby_agg(["dept"], {"salary": ["sum", "mean", "max"]})

# Supported: sum, mean, min, max, count, nunique, std, var
```

### Joins

```python
# Inner join
hf1.join(hf2, on="id")

# Left join
hf1.join(hf2, on="id", how="left")

# Multi-column join
hf1.join(hf2, on=["id", "date"])
```

### Set Operations

```python
hf1.union(hf2)           # UNION ALL
hf1.union(hf2, all=False) # UNION (distinct)
hf1.intersect(hf2)       # INTERSECT
hf1.except_(hf2)         # EXCEPT
```

### Data Cleaning

```python
# Fill missing values
hf.fillna(0)
hf.fillna({"age": 0, "name": "Unknown"})

# Drop rows with missing values
hf.dropna()
hf.dropna(subset=["age", "salary"])

# Replace values
hf.replace({"old_value": "new_value"})
hf.replace({"col1": {1: 10, 2: 20}})

# Clip values
hf.clip(lower=0, upper=100)
hf.clip(lower={"age": 0}, upper={"age": 150})
```

### Feature Engineering

```python
# Add new columns
hf.assign(salary_k=lambda df: df["salary"] / 1000)

# One-hot encoding
hf.one_hot_encode(["dept"])

# Type casting
hf.astype({"age": "int64", "salary": "float64"})

# Cumulative operations
hf.cumsum()
hf.cumprod()
hf.cummin()
hf.cummax()
hf.diff()
```

### Time Series

```python
# Forward fill
hf.time_series_impute("ffill")

# Backward fill
hf.time_series_impute("bfill")

# With datetime column
hf.time_series_impute("ffill", datetime_col="date")
```

### SQL Passthrough

```python
result = hf.sql("""
    SELECT dept, COUNT(*) as count, AVG(salary) as avg_salary
    FROM self
    WHERE age > 25
    GROUP BY dept
    HAVING COUNT(*) > 1
""")
```

### Streaming & Out-of-Core

```python
# Process in chunks
for chunk in hf.fetch_chunked(batch_size=10000):
    process(chunk)

# Arrow streaming
reader = hf.to_arrow_reader(batch_size=8192)
for batch in reader:
    process_arrow(batch)

# PyTorch DataLoader
dataloader = hf.to_torch_dataloader(target_column="salary", batch_size=32)
```

### File I/O

```python
# Read
hf = HybridFrame.from_csv("data.csv")
hf = HybridFrame.from_parquet("data.parquet")

# Write
hf.write_csv("output.csv")
hf.write_parquet("output.parquet")

# With options
hf.write_csv("output.csv", index=False)
```

## Engine Selection

HybridFrame automatically selects the best engine for each operation:

| Operation | Engine | Why |
|-----------|--------|-----|
| `filter()` | DuckDB | SQL-native, lazy evaluation |
| `select()` | Pandas | Zero-copy metadata |
| `sort_values()` | DuckDB | Efficient top-K with heap |
| `groupby_agg()` | DuckDB | Native aggregation |
| `join()` | DuckDB | Hash join, cross-connection transfer |
| `head()` / `tail()` | Pandas | Immediate materialization |
| `rename()` / `drop()` | Pandas | Zero-copy metadata |
| `assign()` | Pandas | ML feature engineering |
| `one_hot_encode()` | Pandas | Direct DataFrame access |

## Memory Management

```python
# Set memory limit
HybridFrame.set_max_memory_gb(4.0)

# Auto-chunking for large materializations
# (automatically chunks when estimated size > memory limit)

# Force materialization (bypass OOM check)
hf.to_pandas(force=True)
```

## Thread Safety

HybridFrame uses a connection pool for thread-safe DuckDB access:

```python
import threading

def process_chunk(chunk):
    hf = HybridFrame.from_pandas(chunk)
    result = hf.filter("value > 0").to_pandas()

threads = [threading.Thread(target=process_chunk, args=(chunk,)) 
           for chunk in data_chunks]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Type Stubs

HybridFrame includes full type stubs for IDE support:

```python
hf: HybridFrame = HybridFrame.from_pandas(df)
result: HybridFrame = hf.filter("age > 25")
df: pd.DataFrame = result.to_pandas()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**mukesh** - [GitHub](https://github.com/shell-bay)
