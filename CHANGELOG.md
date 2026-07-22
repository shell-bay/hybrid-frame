# Changelog

## [0.3.0] - 2026-07-22
### Added
- show_plan() method for query plan debugging
- O(n) complexity documentation on all scanning methods
- Comprehensive pytest suite (186 tests)
- Connection pool with acquire/release
- Streaming fetch_chunked, to_arrow_reader, to_torch_dataloader
- DuckDB-native fillna, isna, dropna, rename, value_counts (no Pandas materialisation)
- One-hot encoding via DuckDB CASE expressions
- Time-series impute via window functions
- memory_usage() method

### Fixed
- Connection pool deadlock: put() → put_nowait()
- Dtypes DESCRIBE column name: data_type → column_type
- pyproject.toml TOML syntax and version sync
- Thread-safety in concurrent pool access
