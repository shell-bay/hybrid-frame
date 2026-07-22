# HybridFrame — Production Readiness Plan

## Architecture Overview

```
HybridFrame
  ├── Engine: DuckDB (default, lazy, out-of-core)
  └── Engine: Pandas (on-demand, materialised, ML-ready)
```

Auto-transition via `@_ensure_engine` decorator. No `.compute()` calls.

---

## Agent 1: Engine & Performance (Core Stability)

**Focus:** Connection lifecycle, memory safety, DuckDB-native ops, thread safety.

### Tasks

| # | Task | Priority | Current State |
|---|------|----------|---------------|
| 1.1 | Connection pool: all connections tracked, clean shutdown | P0 | Some connections leak on error paths |
| 1.2 | `_temp_view` → `CREATE OR REPLACE TEMP VIEW` to avoid DROP race | P0 | DROP VIEW can fail if view already dropped |
| 1.3 | Add `distinct()` method (DuckDB-native) | P1 | Missing |
| 1.4 | Add `sample()` method (DuckDB `USING SAMPLE`) | P1 | Missing |
| 1.5 | Add `union()` / `intersect()` / `except_()` (DuckDB set ops) | P1 | Missing |
| 1.6 | Add `corr()` / `cov()` aggregation support | P1 | Missing |
| 1.7 | Add `quantile()` / `median()` DuckDB paths | P1 | Missing |
| 1.8 | `_estimate_relation_memory` — use EXPLAIN cardinality before `shape[0]` | P1 | Falls back to full COUNT |
| 1.9 | Add `clip()` method (COALESCE with upper/lower bounds) | P2 | Missing |
| 1.10 | Add `astype()` — DuckDB CAST projection | P2 | Missing |
| 1.11 | `filter()`  → support list of conditions | P2 | Single string only |
| 1.12 | `select()` → support string column (single) | P2 | List only |

### Verification
- 186 existing tests still pass
- Thread safety test: 0 connection-closed errors
- Connection pool: no leak after 1000 creates/closes

---

## Agent 2: API Expansion & Pandas Parity (Feature Complete)

**Focus:** Missing methods, ergonomics, feature engineering depth.

### Tasks

| # | Task | Priority | Current State |
|---|------|----------|---------------|
| 2.1 | Add `replace()` — DuckDB `REPLACE` or Pandas `.replace()` | P1 | Missing |
| 2.2 | Add `map()` — column value mapping via CASE/Dict | P1 | Missing |
| 2.3 | Add `where()` — conditional replace (DuckDB `CASE` or `np.where`) | P1 | Missing |
| 2.4 | Add `between()` — column filter via BETWEEN | P1 | Missing |
| 2.5 | Add `isnull()` / `notnull()` aliases | P1 | Missing |
| 2.6 | Add `idxmin()` / `idxmax()` | P1 | Missing |
| 2.7 | Add `abs()` / `round()` column ops | P2 | Missing |
| 2.8 | Add `diff()` — DuckDB `LAG - value` window | P2 | Missing |
| 2.9 | Add `pct_change()` — DuckDB window | P2 | Missing |
| 2.10 | Add `cumsum()` / `cumprod()` / `cummin()` / `cummax()` | P2 | Missing |
| 2.11 | Add `rolling()` — window frame with aggregations | P2 | Missing |
| 2.12 | Add `resample()` — date bucketing via `DATE_TRUNC` | P2 | Missing |
| 2.13 | Better `__getitem__` — support slice, boolean list, callable | P2 | String / list only |
| 2.14 | Add `__setitem__` — column assignment | P2 | Missing |
| 2.15 | Add `pop()` — extract and drop column | P2 | Missing |
| 2.16 | Add `insert()` — insert column at position | P2 | Missing |
| 2.17 | Add `assign()` — add/modify columns (Pandas-like) | P2 | Missing |
| 2.18 | Add `query()` — filter using Pandas-style query syntax | P2 | Missing |

### Verification
- 186 existing tests still pass
- New tests for each added method
- Dual engine path tested (DuckDB + Pandas)

---

## Agent 3: Testing, CI & Polish (Production Hardening)

**Focus:** Test coverage, property-based tests, CI, benchmarks, performance.

### Tasks

| # | Task | Priority | Current State |
|---|------|----------|---------------|
| 3.1 | Add GitHub Actions CI (test on 3.9–3.13, DuckDB stable) | P0 | Missing |
| 3.2 | Add Makefile with `test`, `test-all`, `bench`, `clean` targets | P1 | Missing |
| 3.3 | Add `.gitignore` (Python project defaults) | P1 | Missing |
| 3.4 | Add property-based tests (Hypothesis) for key methods | P1 | Missing |
| 3.5 | Add 1M-row stress test (scale test) | P1 | Missing |
| 3.6 | Add benchmark suite (`pytest-benchmark`) | P1 | Missing |
| 3.7 | Add type stubs (`py.typed` + `hybrid_frame.pyi`) | P1 | Missing |
| 3.8 | Add pre-commit config (black, ruff, mypy) | P2 | Missing |
| 3.9 | Review all docstrings for NumPy-style completeness | P2 | Some missing |
| 3.10 | Add `__all__` to control public API exports | P2 | Missing |
| 3.11 | Add CHANGELOG.md | P2 | Missing |
| 3.12 | Verify all errors produce `HybridFrameError` (not raw DuckDB errors) | P1 | Most done |
| 3.13 | Add `ruff` / `mypy` config to pyproject.toml | P2 | Missing |
| 3.14 | Add `__sizeof__()` method for `sys.getsizeof` support | P2 | Missing |
| 3.15 | Ensure `copy()` returns same type (not `HybridFrame` hardcoded) | P1 | Uses `self.__class__` |

### Verification
- CI green on all Python versions
- Hypothesis: no counterexample found in 10k runs
- Benchmarks: within 10% of expected targets
- `ruff check .` — zero violations
- `mypy .` — zero errors

---

## Timeline

```
Phase 1 (Agents 1+2) — Feature + Engine
  ├── Agent 1: Core engine fixes (P0/P1 items)
  ├── Agent 2: Panda parity methods (P1 items)
  └── All tests pass

Phase 2 (Agent 3) — Testing & CI
  ├── CI pipeline + Makefile
  ├── Property-based tests (Hypothesis)
  ├── Benchmarks
  └── All tests pass (186 + new)

Phase 3 (All) — Polish
  ├── ruff / mypy clean
  ├── Docstring audit
  ├── Type stubs
  └── Final review
```

---

## Non-Goals (Out of Scope for This Round)

- PyPI release (handled separately)
- Documentation site / Sphinx
- Polars backend
- Distributed / Dask integration
- GPU acceleration (cuDF)
