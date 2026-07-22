# HybridFrame Audit Report

Generated: 2026-07-22
Scope: Full read-only code review of `hybrid_frame.py` (2047 lines), test suites, benchmarks, and build config.
Tests: 285 pytest + 54 legacy + 7 hypothesis — all green at HEAD.

---

## 1. Performance

### 1.1 Unnecessary Materialisations (P0)

| Location | Issue | Impact |
|---|---|---|
| `__getitem__` single column (line 485) | Calls `.df()` on the relation just to return one column via DuckDB `select()`. Every `hf["col"]` on a DuckDB engine triggers a full materialisation. | O(n) materialisation for O(1) projection |
| `_getitem_callable` (line 521) | Always calls `self._relation.df()` to materialise the entire relation before applying the callable mask. | O(n) memory for what could be a SQL subquery |
| `_getitem_boolean` (line 531) | Same pattern — `self._relation.df()` materialises everything. | O(n) memory |
| `__getitem__` multi-column (line 488) | Returns `pd.DataFrame` via `select().df()`, always materialised. | O(n) memory |
| `head(n)` DuckDB path (line 713) | Returns `pd.DataFrame` via `limit(n).df()`. Fine for head, but the returned DataFrame is disconnected from the engine — user cannot chain lazy ops after. | Inconsistent API (head returns pd.DataFrame, filter returns HybridFrame) |
| `tail(n)` fallback (line 731) | When `tail()` isn't natively supported, runs `COUNT(*)` + full-scan `LIMIT/OFFSET` via `.df()`. | Double scan |
| `sql()` (line 909) | **Always** calls `.df()` and sets engine to Pandas. SQL queries that could remain lazy are eagerly materialised. | Forces RAM materialisation even for read-only queries. User must re-create a HybridFrame to chain more lazy ops. |
| `rename()` (line 1088), `fillna()` (line 1136), `clip()` (line 1183), `astype()` (line 1217), `replace()` (line 1570), `where()` (line 1600), `between()` (line 1619), `abs()` (line 1673), `round()` (line 1692), `diff()` (line 1724), cumulative ops (line 1769) | These use `self._conn.sql(...)` which materialises the relation result through the DuckDB SQL API, losing the DuckDB relation API chain. Compare with `filter()` which uses `self._relation.filter()` — the native `DuckDBPyRelation` method stays lazy. | Every transformation via `conn.sql()` breaks the lazy chain: the result is a fresh materialised relation, not a pipelined one. Adds redundant planning overhead. |
| `to_pandas(copy=True)` (line 1348) | When already in Pandas engine, always deep-copies even if caller will discard the original after. | 2× memory temporarily |

**Recommendation:** Use `self._relation = self._relation.select(...)` / `.filter(...)` for native chaining where possible. For column-expansion ops (fillna, astype, etc.), prefer `rel.project()` if available, or accept the limitation but add `@_ensure_engine(Engine.DUCKDB_RELATION)` decorator-consistent patterns.

### 1.2 Missing DuckDB-Native Optimisations (P1)

| Method | Current | DuckDB-Native Alternative |
|---|---|---|
| `isna()` | `SELECT col IS NULL AS col` → `.df()` (materialised boolean DataFrame) | Can use `rel.select(...)` with expressions if DuckDBPyRelation supports computed columns natively |
| `nunique()` | `COUNT(DISTINCT col)` via `.conn.sql()` → `.df()` | Use `rel.aggregate()` if available instead of raw SQL |
| `value_counts()` | `GROUP BY ... ORDER BY` via `.conn.sql()` → `.df()` | Same — prefer `rel.aggregate()` |
| `dropna()` | `SELECT * WHERE NOT (...)` via `.conn.sql()` | Can use `rel.filter()` with a constructed condition string |
| `idxmin`/`idxmax` | `ROW_NUMBER()` window via `.conn.sql()` → `.df()` | Could be `rel.argmin()`/`argmax()` if DuckDBPyRelation supports it |
| `copy()` DuckDB path (line 415) | Shares the same `self._relation` reference (shallow). Mutations on the copy affect the original. | Need `self._relation = self._relation.union(None)` or `rel.set_alias()` to break aliasing |

### 1.3 `__getitem__` Slice Performance (P1)

- Line 511-515: For non-zero-start slices on DuckDB, creates a temp view and runs `SELECT * FROM v LIMIT n OFFSET start`. This materialises via `.df()` and then wraps in a new `HybridFrame.from_pandas(result)`. 
- For zero-start slices, correctly uses `self._relation.limit(n)` — lazy. Inconsistent with non-zero-start path.

### 1.4 Benchmark Gaps

- No benchmarks for: `join`, `sql`, `time_series_impute`, `one_hot_encode`, `fetch_chunked`, `to_arrow_reader`, `to_torch_dataloader`, `describe`, `dropna`, `fillna`, `replace`, `where`, `between`, `diff`, cumulative ops.
- Benchmarks only test with 100K rows — no multi-GB or OOM-threshold datasets.

---

## 2. Security

### 2.1 SQL Injection Vectors (P0)

Every method that accepts user-controlled strings and interpolates them into SQL is a vector. The `_sql_literal()` helper (line 164) provides correct escaping for **values** but **identifiers (column names) are NOT escaped**.

#### 2.1.1 Unquoted Identifiers in SQL (Critical)

| Method | Line | Pattern | Risk |
|---|---|---|---|
| `sort_values` | 632 | `f'"{c}" {order}'` — quoted, safe for identifiers | **Low** (quoted) |
| `groupby_agg` | 750-755 | `f'"{c}"'` — quoted | **Low** (quoted) |
| `join` | 794 | `f'l."{k}" = r."{k}"'` — quoted | **Low** (quoted) |
| `filter` | 606 | `self._relation.filter(condition)` — uses DuckDB's parameterised API | **Low** (delegates to DuckDB) |
| `one_hot_encode` col_name | 1036 | `col_name = str(v).replace("'", "").replace("\"", "")` — custom sanitisation | **Medium** — the `_sql_literal(v)` is used for the comparison value, safe. But the column alias `col_name` has weak sanitisation (only strips quotes/spaces). |
| `select` | 622 | `self._relation.select(*columns)` — DuckDB relation API | **Low** (native API) |
| `value_counts` | 1264 | `f'"{column}"'` — quoted | **Low** |
| `time_series_impute` datetime_col | 971-973 | `f'"{datetime_col}"'` — quoted | **Low** |

#### 2.1.2 User-Controlled Expression Injection (Critical)

| Method | Line | Pattern | Risk |
|---|---|---|---|
| `filter` | 606 | `self._relation.filter(condition)` | **Medium** — DuckDB's `.filter()` accepts arbitrary SQL expressions. If a user passes `"1=1; DROP TABLE ..."`, it could be dangerous (though DuckDB doesn't have `DROP TABLE` on registered views easily). |
| `sql` | 907 | `query.replace("self", view_name)` then `self._conn.sql(resolved)` | **High** — Full arbitrary SQL execution. The `self` → `view` replacement is naive: a malicious query with `self` in a string literal would be corrupted, and there's no privilege separation. Documented as "run arbitrary SQL" so partly by design. |
| `where` | 1590 | `CASE WHEN ({cond}) THEN ...` — `cond` is a raw string interpolated into SQL | **High** — `cond` is a user-supplied expression, directly interpolated. If a user passes `cond = "a > 0; SELECT 1"` it's safe within CASE WHEN syntax, but complex injections are possible. |
| `astype` | 1212 | `CAST("col" AS {target})` — `target` is a user-supplied DuckDB type string | **Medium** — DuckDB type strings like `BIGINT` are safe, but injection via `BIGINT); SELECT 1; --` might be possible. |
| `clip` | 1174-1178 | `GREATEST("col", {_sql_literal(lo)})` — values are escaped | **Low** |
| `between` | 1615-1616 | `f'"{column}" BETWEEN {_sql_literal(low)} AND {_sql_literal(high)}'` — column quoted, values escaped | **Low** |
| `drop` | 1107 | `self.select(keep)` — uses existing column names | **Low** (no injection) |
| `rename` | 1085-1086 | `f'"{c}" AS "{alias}"'` — both quoted | **Low** |

### 2.2 `_csv_kwargs_to_sql` Injection Vector (P1)

Line 1845-1856: Converts kwargs to SQL options for `read_csv_auto`. If a user passes `delimiter="'; DROP TABLE ...;'"`:

```python
# User-supplied strings are single-quoted:
parts.append(f"{k}='{v}'")  # line 1853
```

A kwarg value containing a single quote would break the SQL string boundary. While the set of allowed CSV kwargs is validated with `_ALLOWED_CSV_KWARGS`, the **values** within those allowed keys are not sanitised.

**Impact:** A caller who controls `csv_kwargs` (e.g., in a web service accepting user config) could inject SQL via delimiter/quotechar/etc.

### 2.3 `_sql_literal()` Coverage Gaps

- `bytes`, `bytearray`, `datetime.date`, `datetime.datetime` (non-pd.Timestamp), `Decimal`, `complex`, `pd.NaT` — not handled explicitly. Falls through to `str(value).replace("'", "''")` which is safe but may produce semantically wrong SQL.
- `np.nan` via `np.floating` check (line 173): returns `"NULL"` — correct.
- `pd.NaT` returns as string `"NaT"` rather than SQL `NULL` — **bug**.

### 2.4 Recommendations

1. **Parameterised queries** — DuckDB supports `conn.execute(sql, params)`. Replace f-string SQL construction with parameterised queries for all value interpolation. This eliminates injection vectors entirely for values.
2. **Identifier whitelisting** — For column names, maintain a whitelist against `self.columns` before interpolation. Already done in `rename`, `drop`, `astype`, `fillna`, `clip` (iterate over `self.columns`) — but not in `sort_values` `by` parameter.
3. **Add `_sql_identifier()` helper** — for column/table names that double-quotes and validates against allowed identifiers.
4. **Document `sql()` and `filter()` security stance** — explicitly state that these methods accept arbitrary SQL and should not be exposed to untrusted input.

---

## 3. Memory

### 3.1 Double-Copy / OOM Risk Analysis

#### 3.1.1 `_estimate_relation_memory` Accuracy (P1)

Lines 144-161: Estimates memory using the DuckDB optimiser's `estimated_cardinality` from EXPLAIN output.

**Issues:**
- Falls back to `rel.shape[0]` (line 151) if regex fails, which triggers a full COUNT scan — potentially O(n) just to estimate.
- Per-row byte estimate is simplistic: 8 bytes for numeric types, 50 bytes for everything else. VARCHAR columns with long strings, nested types (LIST, STRUCT), or JSON columns will be wildly underestimated.
- `HUGEINT` (16 bytes) is counted as 8.
- DuckDB DECIMAL with variable precision/scale is always 8 bytes.
- Does not account for DuckDB's internal compression or columnar storage.

#### 3.1.2 `_to_pandas()` Materialisation Path (P0)

Line 587: `self._df = self._relation.df()` — this creates a Pandas DataFrame via Arrow zero-copy. However:
- After assignment, `self._relation = None` (line 588).
- **If the DuckDB relation was using a temporary view or complex CTE, the memory is double-allocated**: once in DuckDB's internal buffer manager, once in Pandas. DuckDB may spill to temp, but Pandas cannot.
- The old relation's memory is only freed when Python GC runs, not immediately.

#### 3.1.3 `to_pandas(copy=True)` Double Copy (P1)

Line 1348: `return self._df.copy() if copy else self._df`

When called on Pandas engine with `copy=True`:
- `self._df` already exists in memory.
- `.copy()` deep-copies every column.
- The caller gets a second copy.
- If the caller then discards the original (e.g., `df = hf.to_pandas(); hf.close()`), the original sits until GC.

**Recommendation:** Document that `copy=False` is available and safe when the caller promises read-only access.

#### 3.1.4 `from_pandas(copy=True)` + `_to_duckdb()` Double Copy (P2)

Line 390: `hf._df = df.copy() if copy else df`
Line 569: `self._relation = self._conn.from_df(self._df)` — DuckDB copies the Pandas data into its own buffer.

If user calls `HybridFrame.from_pandas(df, copy=True)` then immediately `._to_duckdb()`:
1. `df.copy()` — deep copy in Pandas.
2. `from_df()` — DuckDB copies (or mmaps) the Pandas data.
3. The `.copy()` in Pandas is wasted — the user's original `df` was already separate.

#### 3.1.5 `fetch_chunked()` — No Memory Guard

Lines 1396-1407: Iterates over data in batches. When the engine is DuckDB:
- Gets a result handle via `conn.execute()`, then calls `fetch_df_chunk(batch_size)`.
- Each chunk is wrapped in `HybridFrame.from_pandas(chunk, copy=False)`.
- **No OOM guard** is applied per chunk — each chunk is materialised directly.

#### 3.1.6 `join()` Cross-Connection Materialisation (P1)

Lines 781-787: When `other._conn is not self._conn`:
- Materialises `other._relation` to Pandas DataFrame (`.df()`).
- Closes `other._conn`.
- Reassigns `other._conn = self._conn`.
- `other._df = df` (Pandas DataFrame).
- Re-registers into DuckDB via `_to_duckdb()`.

This is a **triple copy**: DuckDB relation → Pandas DataFrame → DuckDB `from_df()`. For large datasets (e.g., 10 GB), this is 30 GB peak memory.

### 3.2 OOM Guard Gaps (P1)

`_warn_if_oom_risk()` is only called from `_to_pandas()` → `to_pandas()`. **Not called from**:
- `__getitem__` (single column, callable, boolean) — all call `.df()` directly without guard.
- `head()` / `tail()` — call `.limit(n).df()` without guard.
- `_getitem_slice()` — non-zero-start path calls `.df()` without guard.
- `describe()` — calls `SUMMARIZE` via `.df()` without guard.
- `sql()` — calls `.df()` without guard.
- `join()` cross-connection path — calls `.df()` without guard.
- `isna()`, `nunique()`, `value_counts()` — all call `.df()` without guard.
- `one_hot_encode()` — calls `.df()` for DISTINCT scan without guard.
- `idxmin`/`idxmax` — call `.df()` without guard.
- `fetch_chunked()` — per-chunk materialisation without guard.
- `_ensure_both_duckdb()` — calls `.df()` without guard.

### 3.3 `psutil` as Optional Dependency (P2)

Line 42-46: If `psutil` is not installed, `_available_ram_gb()` returns 999.0 and `_total_ram_gb()` returns 999.0. This means the OOM guard is completely disabled without `psutil`. 

`psutil` is listed as an optional dependency under `[project.optional-dependencies] memory-guard`, but:
- Most users won't install the `memory-guard` extra.
- The memory guard silently becomes a no-op with no warning.

---

## 4. Reliability

### 4.1 Connection Pool (P1)

| Issue | Detail |
|---|---|
| `_POOL_MAX_SIZE = 16` (line 246) | Queue maxsize is 16, but `_leased_connections` is unbounded. `acquire_connection()` creates new connections when the pool is empty, leading to unlimited total connections across all instances. |
| Connection leak in `__init__` (line 316) | When `memory_limit`, `temp_directory`, or `threads` are provided, a **new** connection is created with `duckdb.connect(config=...)` (line 328), bypassing the pool entirely. This connection is never pooled — `close()` calls `release_connection()` which tries to return it to the pool, but the pool (`queue.Queue`) has no knowledge of this connection and it may fail. |
| Thread safety | `_pool_lock` protects `_pool` and `_leased_connections`, but `acquire_connection()` calls `conn.execute("SELECT 1")` inside the lock — a blocking I/O operation under a lock. Not a correctness bug but reduces concurrency under contention. |
| `other._conn.close()` in join (line 783) | Silently closes the other frame's connection without notifying the pool. The connection is never returned to the pool, and `_leased_connections` retains a stale reference. |

### 4.2 Error Consistency

| Issue | Detail |
|---|---|
| DuckDB exceptions | Most methods wrap `duckdb.Error` in `HybridFrameError`, but `__getitem__` (line 486-488), `head` (line 713), `tail` (line 726-733), `isna`, `nunique`, `value_counts`, `idxmin`, `idxmax`, `describe` do **not** wrap. Raw DuckDB exceptions can propagate to users. |
| `FileNotFoundError` | `from_csv` and `from_parquet` raise `FileNotFoundError` (stdlib), not `HybridFrameError`. |
| `__getitem__` single column DuckDB path (line 486) | If column doesn't exist, raises `duckdb.Binder` exception directly. |
| `show_plan` (line 929) | Raises `HybridFrameError` with a helpful message — good. |
| `_csv_kwargs_to_sql` / `_pq_kwargs_to_sql` | No validation guards against runtime SQL errors. |

### 4.3 Thread Safety

| Issue | Detail |
|---|---|
| `HybridFrame` instances are **not thread-safe** | No locks protect `self._relation`, `self._df`, `self._engine`, `self._conn`. Concurrent access from multiple threads on the same instance will cause races. |
| `_ensure_engine` decorator | If two threads call DuckDB and Pandas methods concurrently, double transition is possible. |
| `close()` during concurrent use | A thread calling `close()` while another uses `self._relation` will get `None` mid-operation. |

These are documented trade-offs for a single-threaded ML library, but worth noting.

### 4.4 `__del__` Safety

Line 408-412: `__del__` calls `self.close()`. If `close()` raises, the exception is swallowed. However, `__del__` is called by the GC in **any** thread, and `release_connection()` acquires `_pool_lock` — a potential issue if the GC runs during interpreter shutdown when the threading module may be partially torn down.

### 4.5 Type Stability

- `dtypes` property (line 458-471): DuckDB path returns `dtype=object` with string type names (e.g., `"BIGINT"`, `"VARCHAR"`). Pandas path returns numpy dtypes. Downstream code expecting numeric dtypes will break when switching engines.
- `memory_usage` (line 1483-1512): DuckDB path returns an estimate with column name `"estimated_memory_usage"`; Pandas path returns actual usage with column name `"Memory usage"`. Inconsistent.

---

## 5. Priority-Ordered Recommendations

### P0 — Critical (security or correctness)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| 1 | **Parameterise all SQL value interpolation** — replace f-string `_sql_literal()` calls with `conn.execute(sql, params)` for all user-supplied values. | 2-3 days | Eliminates SQL injection for values. |
| 2 | **Add OOM guard to every `.df()` call site** — wrap all materialisations in `_warn_if_oom_risk()`. | 1 day | Prevents OOM crashes from non-`to_pandas()` paths. |
| 3 | **Fix cross-connection join materialisation** — avoid Triple Copy by registering the other frame's relation on the same connection without materialising to Pandas first. | 0.5 day | Huge memory savings for join-heavy workflows. |
| 4 | **Add `_sql_identifier()` for column names** — double-quote and validate against `self.columns` whitelist. | 0.5 day | Prevents identifier injection. |

### P1 — High (performance or reliability)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| 5 | **Keep `sql()` result lazy** — add a `lazy=True` parameter that keeps the result as a DuckDB relation instead of materialising to Pandas. | 1 day | Enables SQL-based lazy chaining. |
| 6 | **Use `rel.project()`/`rel.aggregate()` instead of `conn.sql()`** for fillna, astype, clip, rename, replace, where, between, abs, round, diff, cumulative ops. | 2-3 days | Keeps lazy chain intact; avoids re-planning cost. |
| 7 | **Fix `_estimate_relation_memory`** — use `rel.types` per-column actual DuckDB storage size, fall back to `rel.explain()` cardinality without scanning. | 1 day | More accurate OOM guard. |
| 8 | **Fix `__getitem__` single-column DuckDB path** — use `self._relation.select(key).fetchone()` or `self._relation.select(key).df()[key]` without materialising all columns. | 0.5 day | Speeds up column access. |
| 9 | **Fix `_getitem_callable` and `_getitem_boolean`** — avoid full materialisation via `rel.filter()` or `rel.select()`. | 1 day | Reduces memory for boolean-indexed access. |
| 10 | **Warn when `psutil` not installed** — log a warning at import time that OOM guard is disabled. | 0.5 day | Prevents false sense of security. |
| 11 | **Fix `copy()` shallow relation alias** — deep-copy DuckDB relations with `rel.union(None)` or `rel.set_alias()`. | 0.5 day | Correctness — mutations on copy shouldn't affect original. |
| 12 | **Wrap all DuckDB errors in `HybridFrameError`** — add try/except to all remaining `.df()`, `.execute()`, `.sql()` calls. | 1 day | Consistent error surface. |

### P2 — Medium (quality of life)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| 13 | **Type-stabilise `dtypes`** — map DuckDB type names to numpy/pandas dtypes so engine switches don't break downstream code. | 1 day | API consistency. |
| 14 | **Connection pool hard limit** — enforce an absolute max on `_leased_connections` (e.g., 64) and close/error above that. | 0.5 day | Prevents connection leak under heavy use. |
| 15 | **Add `fetch_df_chunk` memory guard** — estimate per-chunk memory and warn if large. | 0.5 day | Safer streaming. |
| 16 | **Improve `head()`/`tail()` return type** — return `HybridFrame` instead of `pd.DataFrame` for consistent chaining. | 1 day | API consistency. |
| 17 | **Handle `pd.NaT` in `_sql_literal()`** — return `NULL` instead of `"NaT"`. | 0.2 day | Correctness. |
| 18 | **Benchmark coverage** — add benchmarks for join, sql, streaming, cumulative ops, groupby_agg with large distinct cardinality. | 1 day | Performance regression detection. |
| 19 | **Document thread-safety guarantees** — explicitly state that HybridFrame is not thread-safe (single-instance). | 0.2 day | User clarity. |

---

## Summary

The library is well-architected with a clean engine-switching design, but has **4 P0 issues** (SQL injection vectors, missing OOM guards, triple-copy join path), **12 P1 issues** (lazy chain breaks, memory estimate inaccuracy, error consistency), and **7 P2 issues** (type stability, connection pool limits, test gaps).

**Top 3 most impactful fixes:**
1. Parameterised SQL queries — eliminates the root cause of injection vectors.
2. OOM guard on all materialisation paths — prevents crashes from non-`to_pandas()` paths.
3. Keep lazy chain intact by using DuckDB relation API instead of `conn.sql()`.
