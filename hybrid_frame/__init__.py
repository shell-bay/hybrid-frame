"""
HybridFrame: A high-performance Python data framework unifying DuckDB and Pandas.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │                  HybridFrame                     │
  │  ┌──────────────┐        ┌───────────────────┐  │
  │  │ DuckDB Engine │◄──────►│  Pandas Engine    │  │
  │  │ (Out-of-core) │  auto  │ (In-memory / ML)  │  │
  │  │  Lazy / SQL   │───────│  Materialized DF   │  │
  │  └──────────────┘ state   └───────────────────┘  │
  │                   machine                         │
  └─────────────────────────────────────────────────┘

Every method is wrapped with @_ensure_engine which transitions
state transparently — no user-facing .compute() calls needed.
"""

from __future__ import annotations

__all__ = ["HybridFrame", "HybridFrameError"]
__version__ = "0.3.0"

import logging
import math
import os
import queue
import re
import sys
import threading
import tracemalloc
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

# opt into future pandas behaviour: fillna/ffill/bfill no longer silently
# downcast object-dtype columns; users should call .infer_objects(copy=False)
# when explicit downcasting is needed
pd.set_option("future.no_silent_downcasting", True)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore[assignment]
    HAS_PYARROW = False

try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Logging setup – structured info to stderr
# ---------------------------------------------------------------------------
logger = logging.getLogger("HybridFrame")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[HybridFrame %(levelname)s] %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class HybridFrameError(Exception):
    """Base exception for all HybridFrame errors."""


# ---------------------------------------------------------------------------
# Engine state enum
# ---------------------------------------------------------------------------
class Engine(Enum):
    DUCKDB_RELATION = "DUCKDB_RELATION"
    PANDAS_DATAFRAME = "PANDAS_DATAFRAME"


# ---------------------------------------------------------------------------
# Memory guard utilities
# ---------------------------------------------------------------------------
_SYSTEM_MEMORY_GB: Optional[float] = None

if HAS_PSUTIL:

    def _available_ram_gb() -> float:
        return psutil.virtual_memory().available / (1024 ** 3)

    def _total_ram_gb() -> float:
        global _SYSTEM_MEMORY_GB
        if _SYSTEM_MEMORY_GB is None:
            _SYSTEM_MEMORY_GB = psutil.virtual_memory().total / (1024 ** 3)
        return _SYSTEM_MEMORY_GB
else:

    def _available_ram_gb() -> float:
        return 999.0

    def _total_ram_gb() -> float:
        return 999.0

    logger.warning(
        "psutil not installed – memory guard disabled. "
        "Install with: pip install psutil"
    )


def _warn_if_oom_risk(estimated_gb: float) -> None:
    if estimated_gb < 0.01:
        return
    avail = _available_ram_gb()
    if estimated_gb > avail:
        logger.error(
            "OOM RISK: ~%.1f GB needed but only %.1f GB available. "
            "Call .to_pandas(force=True) to override.",
            estimated_gb,
            avail,
        )
        raise MemoryError(
            f"Estimated memory {estimated_gb:.1f} GB exceeds available "
            f"{avail:.1f} GB. Pass force=True to bypass."
        )
    if estimated_gb > avail * 0.6:
        logger.warning(
            "Large materialisation: ~%.1f GB of %.1f GB available. "
            "System may slow down.",
            estimated_gb,
            avail,
        )


_NUMERIC_TYPE_HINTS = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "FLOAT", "DOUBLE", "DECIMAL", "REAL",
}

_DUCKDB_TO_NUMPY_TYPE = {
    "BIGINT": "int64",
    "INTEGER": "int32",
    "SMALLINT": "int16",
    "TINYINT": "int8",
    "HUGEINT": "object",
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "REAL": "float32",
    "DECIMAL": "float64",
    "BOOLEAN": "bool",
    "VARCHAR": "object",
    "DATE": "object",
    "TIMESTAMP": "object",
    "TIMESTAMP WITH TIME ZONE": "object",
    "TIME": "object",
    "BLOB": "object",
    "INTERVAL": "object",
    "UUID": "object",
    "JSON": "object",
    "STRUCT": "object",
    "LIST": "object",
    "MAP": "object",
    "ENUM": "object",
    "UNION": "object",
    "BIT": "object",
}

_DUCKDB_TYPE_STORAGE_BYTES = {
    "BOOLEAN": 1,
    "TINYINT": 1,
    "SMALLINT": 2,
    "INTEGER": 4,
    "BIGINT": 8,
    "HUGEINT": 16,
    "FLOAT": 4,
    "DOUBLE": 8,
    "REAL": 4,
    "DECIMAL": 16,
    "DATE": 4,
    "TIME": 8,
    "TIMESTAMP": 8,
    "TIMESTAMP_S": 8,
    "TIMESTAMP_MS": 8,
    "TIMESTAMP_NS": 8,
    "TIMESTAMP WITH TIME ZONE": 8,
    "INTERVAL": 16,
    "UUID": 16,
}


def _estimate_relation_memory(rel: duckdb.DuckDBPyRelation) -> float:
    try:
        plan = rel.explain()
        match = re.search(r'estimated_cardinality[:\s]+(\d+)', plan, re.IGNORECASE)
        if match:
            n_rows = int(match.group(1))
        else:
            n_rows = rel.shape[0]
        col_types = [str(t) for t in rel.types]
    except Exception:
        return 0.0
    if n_rows == 0:
        return 0.0
    bytes_per_row = 0
    for t in col_types:
        base = t.upper().split("(")[0].split("[")[0].strip()
        bytes_per_row += _DUCKDB_TYPE_STORAGE_BYTES.get(base, 50)
    return n_rows * bytes_per_row / (1024 ** 3)


def _sql_literal(value: Any) -> str:
    """Convert a Python value to a DuckDB SQL literal string."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and (math.isnan(float(value)) or math.isinf(float(value))):
        return "NULL" if math.isnan(float(value)) else str(float(value))
    if isinstance(value, np.floating):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, pd.Timestamp):
        return f"TIMESTAMP '{value}'"
    if isinstance(value, pd._libs.NaTType):
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# Memory tracing utilities (opt-in via HYBRIDFRAME_MEMORY_TRACE env var)
# ---------------------------------------------------------------------------
_MEMORY_TRACE = os.environ.get("HYBRIDFRAME_MEMORY_TRACE", "").lower() in ("1", "true", "yes")


class MemoryTracker:
    """Context manager that logs peak memory delta for an operation.

    Only active when ``HYBRIDFRAME_MEMORY_TRACE=1`` is set in the environment.
    Uses ``tracemalloc`` to capture Python-side allocations (does not include
    DuckDB/Arrow off-heap buffers).
    """

    def __init__(self, operation: str, detail: str = "", logger: logging.Logger = None):
        self._op = operation
        self._detail = detail
        self._log = logger or logger
        self._snap_before = None
        self._peak_before = 0

    def __enter__(self) -> "MemoryTracker":
        if _MEMORY_TRACE and tracemalloc.is_tracing():
            self._snap_before = tracemalloc.take_snapshot()
            self._peak_before = tracemalloc.get_traced_memory()[1]
        return self

    def __exit__(self, *args: Any) -> None:
        if not _MEMORY_TRACE or not tracemalloc.is_tracing() or self._snap_before is None:
            return
        snap_after = tracemalloc.take_snapshot()
        peak_after = tracemalloc.get_traced_memory()[1]
        stats = snap_after.compare_to(self._snap_before, "lineno")
        delta = sum(s.size_diff for s in stats)
        peak_delta = peak_after - self._peak_before
        msg = f"MemTrace [{self._op}] delta={delta / 1024 ** 2:.2f} MB  peak_delta={peak_delta / 1024 ** 2:.2f} MB"
        if self._detail:
            msg += f"  ({self._detail})"
        logger.info(msg)


# ---------------------------------------------------------------------------
# The automatic engine-transition decorator
# ---------------------------------------------------------------------------
def _ensure_engine(target_engine: Engine) -> Callable:
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self: "HybridFrame", *args: Any, **kwargs: Any) -> Any:
            if self._engine is target_engine:
                return method(self, *args, **kwargs)
            if target_engine is Engine.DUCKDB_RELATION:
                self._to_duckdb()
            else:
                self._to_pandas()
            return method(self, *args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Helper: context manager for temporary DuckDB SQL views (materialising ops)
# ---------------------------------------------------------------------------
class _temp_view:
    """Register a DataFrame/relation as a temporary view for immediate queries."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        obj: Any,
        view_name: str = "_hf_tmp",
    ) -> None:
        self._conn = conn
        self._obj = obj
        self._view_name = view_name

    def __enter__(self) -> str:
        self._conn.register(self._view_name, self._obj)
        return self._view_name

    def __exit__(self, *exc: Any) -> None:
        try:
            self._conn.execute(f"DROP VIEW IF EXISTS {self._view_name}")
        except Exception:
            pass


# ===================================================================
# HybridFrame
# ===================================================================
class HybridFrame:
    """A dual-engine DataFrame that transparently shifts between
    DuckDB (lazy / out-of-core) and Pandas (materialised / in-memory).

    Basic usage
    -----------
    >>> hf = HybridFrame.from_csv("data.csv")
    >>> hf.filter("age > 21").select(["name", "salary"]) \\
    ...    .groupby_agg(["name"], {"salary": "sum"}) \\
    ...    .reshape("pivot", ...)   # auto → Pandas
    >>> X, y = hf.to_ml_ready("target")
    """

    # -- class-level connection pool ------------------------------------
    _pool: "queue.Queue[duckdb.DuckDBPyConnection]" = queue.Queue()
    _POOL_MAX_SIZE: int = 16
    _leased_connections: "set[duckdb.DuckDBPyConnection]" = set()
    _pool_lock = threading.Lock()
    _pool_lock_inner = threading.Lock()

    # -- memory limit ---------------------------------------------------
    _MAX_MEMORY_GB: float = 0.0

    @classmethod
    def set_max_memory_gb(cls, limit: float) -> None:
        cls._MAX_MEMORY_GB = limit

    @classmethod
    def acquire_connection(cls) -> duckdb.DuckDBPyConnection:
        """Return a live connection from the pool, or create a new one."""
        with cls._pool_lock:
            while not cls._pool.empty():
                conn = cls._pool.get_nowait()
                try:
                    conn.execute("SELECT 1")
                    cls._leased_connections.add(conn)
                    return conn
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    continue
            conn = duckdb.connect()
            cls._leased_connections.add(conn)
            return conn

    @classmethod
    def release_connection(cls, conn: duckdb.DuckDBPyConnection) -> None:
        """Return a connection to the pool, or close it if the pool is full."""
        if conn is None:
            return
        with cls._pool_lock:
            cls._leased_connections.discard(conn)
            try:
                cls._pool.put_nowait(conn)
            except queue.Full:
                try:
                    conn.close()
                except Exception:
                    pass

    @classmethod
    def close_all_connections(cls) -> None:
        """Drain the pool and close every leased connection."""
        with cls._pool_lock:
            while not cls._pool.empty():
                try:
                    conn = cls._pool.get_nowait()
                    conn.close()
                except Exception:
                    pass
            for conn in list(cls._leased_connections):
                try:
                    conn.close()
                except Exception:
                    pass
            cls._leased_connections.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        memory_limit: Optional[str] = None,
        temp_directory: Optional[str] = None,
        threads: Optional[int] = None,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        if connection is not None:
            self._conn = connection
        else:
            self._conn = None
        self._init_config: Dict[str, str] = {}
        if memory_limit is not None:
            self._init_config["memory_limit"] = memory_limit
        if temp_directory is not None:
            self._init_config["temp_directory"] = temp_directory
        if threads is not None:
            self._init_config["threads"] = str(threads)
        self._engine: Engine = Engine.DUCKDB_RELATION
        self._relation: Optional[duckdb.DuckDBPyRelation] = None
        self._df: Optional[pd.DataFrame] = None

    # -- class constructors -------------------------------------------

    @classmethod
    def from_csv(cls, file_path: Union[str, Path], **csv_kwargs: Any) -> "HybridFrame":
        """Stream a CSV file lazily via DuckDB without loading into RAM."""
        path = cls._resolve_path(file_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        _validate_csv_kwargs(**csv_kwargs)
        logger.info("Streaming CSV from %s via DuckDB ...", path)
        hf = cls()
        if hf._conn is None:
            hf._conn = cls.acquire_connection()
        extra = _csv_kwargs_to_sql(**csv_kwargs)
        try:
            hf._relation = hf._conn.sql(f"SELECT * FROM read_csv_auto('{path}'{extra})")
        except duckdb.Error as e:
            raise HybridFrameError(f"Failed to read CSV '{path}': {e}") from e
        hf._engine = Engine.DUCKDB_RELATION
        return hf

    @classmethod
    def from_parquet(cls, file_path: Union[str, Path], **pq_kwargs: Any) -> "HybridFrame":
        """Stream a Parquet file lazily via DuckDB."""
        path = cls._resolve_path(file_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Parquet file not found: {path}")
        _validate_pq_kwargs(**pq_kwargs)
        logger.info("Streaming Parquet from %s via DuckDB ...", path)
        hf = cls()
        if hf._conn is None:
            hf._conn = cls.acquire_connection()
        extra = _pq_kwargs_to_sql(**pq_kwargs)
        try:
            hf._relation = hf._conn.sql(f"SELECT * FROM read_parquet('{path}'{extra})")
        except duckdb.Error as e:
            raise HybridFrameError(f"Failed to read Parquet '{path}': {e}") from e
        hf._engine = Engine.DUCKDB_RELATION
        return hf

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        copy: bool = True,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> "HybridFrame":
        """Wrap an existing Pandas DataFrame in HybridFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame.
        copy : bool
            If True (default), deep-copy the input DataFrame. Set to False
            to share memory when the caller guarantees no mutation.
        connection : duckdb.DuckDBPyConnection, optional
            Reuse an existing DuckDB connection.  When omitted the instance
            acquires one from the class-level pool.
        """
        hf = cls(connection=connection)
        with MemoryTracker("from_pandas", f"shape={df.shape} copy={copy}"):
            hf._df = df.copy() if copy else df
            hf._engine = Engine.PANDAS_DATAFRAME
            logger.info("Wrapped Pandas DataFrame (shape=%s) in HybridFrame.", df.shape)
            return hf

    @staticmethod
    def _resolve_path(file_path: Union[str, Path]) -> str:
        return os.path.abspath(os.path.expanduser(str(file_path)))

    def close(self) -> None:
        """Drop local references and return the connection to the class pool."""
        self._relation = None
        self._df = None
        self._engine = Engine.DUCKDB_RELATION
        if self._conn is not None:
            self.release_connection(self._conn)
            self._conn = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def copy(self) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            hf = self.__class__()
            hf._relation = self._relation
            hf._engine = Engine.DUCKDB_RELATION
            hf._conn = self._conn
            return hf
        return self.__class__.from_pandas(self.to_pandas(copy=False))

    def __repr__(self) -> str:
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            return (
                f"<HybridFrame engine=DuckDB "
                f"shape={self._relation.shape} "
                f"columns={self._relation.columns}>"
            )
        if self._engine is Engine.PANDAS_DATAFRAME and self._df is not None:
            return (
                f"<HybridFrame engine=Pandas "
                f"shape={self._df.shape} "
                f"columns={list(self._df.columns)}>"
            )
        return "<HybridFrame engine=None>"

    def __len__(self) -> int:
        return self.shape[0]

    # ------------------------------------------------------------------
    # Properties — read-only, no engine transition
    # ------------------------------------------------------------------
    @property
    def columns(self) -> List[str]:
        if self._engine is Engine.PANDAS_DATAFRAME and self._df is not None:
            return list(self._df.columns)
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            return list(self._relation.columns)
        return []

    @property
    def shape(self) -> Tuple[int, int]:
        if self._engine is Engine.PANDAS_DATAFRAME and self._df is not None:
            return self._df.shape
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            return self._relation.shape
        return (0, 0)

    @property
    def dtypes(self) -> pd.Series:
        if self._engine is Engine.PANDAS_DATAFRAME and self._df is not None:
            return self._df.dtypes
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            with _temp_view(self._conn, self._relation, "_hf_dtypes") as v:
                result = self._conn.sql(
                    f"SELECT column_name, column_type FROM (DESCRIBE {v})"
                ).df()
                raw_types = result["column_type"].values
                np_types = []
                for t in raw_types:
                    base = t.upper().split("(")[0].split("[")[0].strip()
                    np_types.append(np.dtype(_DUCKDB_TO_NUMPY_TYPE.get(base, "object")))
                return pd.Series(
                    np_types,
                    index=result["column_name"],
                )
        return pd.Series(dtype=object)

    # ------------------------------------------------------------------
    # Column accessor
    # ------------------------------------------------------------------
    def __getitem__(self, key: Union[str, List[str], slice, Callable, List[bool]]) -> Union[pd.Series, pd.DataFrame, "HybridFrame"]:
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if callable(key):
            return self._getitem_callable(key)
        if isinstance(key, (list, np.ndarray)):
            if len(key) > 0 and isinstance(key[0], (bool, np.bool_)):
                return self._getitem_boolean(key)
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if isinstance(key, str):
                try:
                    selected = self._relation.select(key)
                    self._materialise_safely(selected)
                    return selected.df()[key]
                except duckdb.Error as e:
                    raise HybridFrameError(f"Column access failed: {e}") from e
            if isinstance(key, list):
                try:
                    selected = self._relation.select(*key)
                    self._materialise_safely(selected)
                    return selected.df()
                except duckdb.Error as e:
                    raise HybridFrameError(f"Column access failed: {e}") from e
        return self.to_pandas()[key]

    def _getitem_slice(self, key: slice) -> "HybridFrame":
        start = key.start or 0
        stop = key.stop
        step = key.step or 1
        if step != 1:
            raise HybridFrameError("Slice step must be 1")
        if start < 0:
            raise HybridFrameError("Negative slice start not supported")
        if stop is None:
            return self.copy()
        n = stop - start
        if n < 0:
            raise HybridFrameError("Slice stop must be greater than start")
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if start == 0:
                hf = HybridFrame()
                hf._relation = self._relation.limit(n)
                hf._engine = Engine.DUCKDB_RELATION
                hf._conn = self._conn
                return hf
            with _temp_view(self._conn, self._relation, "_hf_slice") as v:
                result = self._conn.sql(
                    f"SELECT * FROM {v} LIMIT {n} OFFSET {start}"
                )
                try:
                    self._materialise_safely(result)
                    pdf = result.df()
                except duckdb.Error as e:
                    raise HybridFrameError(f"Slice failed: {e}") from e
            return HybridFrame.from_pandas(pdf)
        self._to_pandas()
        return HybridFrame.from_pandas(self._df.iloc[start:stop:step])

    def _getitem_callable(self, key: Callable) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            try:
                self._materialise_safely(self._relation)
                pdf = self._relation.df()
            except duckdb.Error as e:
                raise HybridFrameError(f"Materialisation for callable index failed: {e}") from e
            mask = key(pdf)
            return HybridFrame.from_pandas(pdf[mask])
        self._to_pandas()
        mask = key(self._df)
        return HybridFrame.from_pandas(self._df[mask])

    def _getitem_boolean(self, key: Union[List[bool], np.ndarray]) -> "HybridFrame":
        bool_list = list(key)
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            try:
                self._materialise_safely(self._relation)
                pdf = self._relation.df()
            except duckdb.Error as e:
                raise HybridFrameError(f"Materialisation for bool index failed: {e}") from e
            return HybridFrame.from_pandas(pdf[bool_list])
        self._to_pandas()
        return HybridFrame.from_pandas(self._df[bool_list])

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise HybridFrameError("Column name must be a string")
        if self._df is None and self._relation is None:
            self._df = pd.DataFrame()
            self._engine = Engine.PANDAS_DATAFRAME
        else:
            self._to_pandas()
        self._df[key] = value

    def pop(self, column: str) -> pd.Series:
        s = self[column].copy()
        self.drop(column)
        return s

    def assign(self, **kwargs: Any) -> "HybridFrame":
        if self._df is None and self._relation is None:
            self._df = pd.DataFrame()
            self._engine = Engine.PANDAS_DATAFRAME
        else:
            self._to_pandas()
        for name, value in kwargs.items():
            if callable(value):
                self._df[name] = value(self._df)
            else:
                self._df[name] = value
        return self

    # ------------------------------------------------------------------
    # Internal state-transition helpers
    # ------------------------------------------------------------------
    def _to_duckdb(self) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION:
            return self
        if self._df is None:
            raise HybridFrameError("No data loaded in HybridFrame.")
        if self._conn is None:
            if self._init_config:
                self._conn = duckdb.connect(config=self._init_config)
            else:
                self._conn = self.acquire_connection()
            if self._MAX_MEMORY_GB > 0:
                try:
                    self._conn.execute(f"SET memory_limit = '{self._MAX_MEMORY_GB}GB'")
                except Exception:
                    pass
        logger.info(
            "Registering Pandas DataFrame (shape=%s) into DuckDB ...",
            self._df.shape,
        )
        self._relation = self._conn.from_df(self._df)
        self._df = None
        self._engine = Engine.DUCKDB_RELATION
        return self

    def _to_pandas(self, force: bool = False) -> "HybridFrame":
        if self._engine is Engine.PANDAS_DATAFRAME:
            return self
        if self._relation is None:
            raise HybridFrameError("No data loaded in HybridFrame.")
        est = _estimate_relation_memory(self._relation)
        max_mem = self._MAX_MEMORY_GB
        if not force and max_mem > 0 and est > max_mem:
            logger.info(
                "Auto-chunking materialisation (%.4f GB > max %.4f GB)...",
                est, max_mem,
            )
            chunks: List[pd.DataFrame] = []
            for chunk_hf in self.fetch_chunked():
                chunks.append(chunk_hf._df)
            self._df = pd.concat(chunks, ignore_index=True)
            self._relation = None
            self._engine = Engine.PANDAS_DATAFRAME
            return self
        with MemoryTracker("_to_pandas", f"est={est:.3f} GB"):
            self._materialise_safely(self._relation, force=force)
            logger.info(
                "Materialising DuckDB relation (%.1f GB estimated) into Pandas RAM ...",
                _estimate_relation_memory(self._relation),
            )
            if HAS_PYARROW:
                try:
                    arrow_table = self._relation.fetch_arrow_table()
                    self._df = arrow_table.to_pandas(zero_copy_only=True)
                except Exception:
                    self._df = self._relation.df()
            else:
                self._df = self._relation.df()
        self._relation = None
        self._engine = Engine.PANDAS_DATAFRAME
        return self

    @staticmethod
    def _materialise_safely(
        rel: duckdb.DuckDBPyRelation, force: bool = False
    ) -> None:
        est = _estimate_relation_memory(rel)
        if not force:
            _warn_if_oom_risk(est)

    # ------------------------------------------------------------------
    # DuckDB-engine methods  (lazy / out-of-core, using native relation API)
    # ------------------------------------------------------------------
    @_ensure_engine(Engine.DUCKDB_RELATION)
    def filter(self, condition: Union[str, List[str]]) -> "HybridFrame":
        """Filter rows by a SQL condition (kept as a lazy DuckDB relation).

        When *condition* is a list of strings they are joined with `` AND ``.

        **Complexity:** O(n) — full scan when materialised.
        """
        if isinstance(condition, list):
            condition = " AND ".join(condition)
        try:
            self._relation = self._relation.filter(condition)
        except duckdb.Error as e:
            raise HybridFrameError(f"Filter condition failed: {e}") from e
        return self

    def select(self, columns: Union[str, Sequence[str]]) -> "HybridFrame":
        if isinstance(columns, str):
            columns = [columns]
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            try:
                self._relation = self._relation.select(*columns)
            except duckdb.Error as e:
                raise HybridFrameError(f"Select columns failed: {e}") from e
            return self
        self._to_pandas()
        missing = [c for c in columns if c not in self._df.columns]
        if missing:
            raise HybridFrameError(f"Columns not found: {missing}")
        self._df = self._df[columns]
        return self

    @_ensure_engine(Engine.DUCKDB_RELATION)
    def sort_values(self, by: Union[str, List[str]], ascending: bool = True) -> "HybridFrame":
        if isinstance(by, str):
            by = [by]
        valid = set(self.columns)
        for c in by:
            if c not in valid:
                raise HybridFrameError(f"Column {c!r} not found in DataFrame columns: {self.columns}")
        order = "ASC" if ascending else "DESC"
        order_expr = ", ".join(f'{_sql_identifier(c)} {order}' for c in by)
        try:
            self._relation = self._relation.order(order_expr)
        except duckdb.Error as e:
            raise HybridFrameError(f"Sort failed: {e}") from e
        return self

    def limit(self, n: int) -> "HybridFrame":
        if n < 0:
            raise HybridFrameError("Limit must be non-negative.")
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            try:
                self._relation = self._relation.limit(n)
            except duckdb.Error as e:
                raise HybridFrameError(f"Limit failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.iloc[:n].copy()
        return self

    @_ensure_engine(Engine.DUCKDB_RELATION)
    def distinct(self) -> "HybridFrame":
        """Remove duplicate rows via ``SELECT DISTINCT *``.

        **Complexity:** O(n log n) — full scan + sort/hash for dedup.
        """
        if self._df is None and self._relation is None:
            return HybridFrame()
        rel = self._relation
        try:
            self._relation = self._conn.sql("SELECT DISTINCT * FROM rel")
        except duckdb.Error as e:
            raise HybridFrameError(f"distinct failed: {e}") from e
        return self

    @_ensure_engine(Engine.DUCKDB_RELATION)
    def sample(
        self,
        n: Union[int, float],
        method: str = "reservoir",
    ) -> "HybridFrame":
        """Sample rows using DuckDB's ``USING SAMPLE`` clause.

        Parameters
        ----------
        n : int or float
            If int, number of rows.  If float (0–100), percentage of rows.
        method : str
            One of ``'bernoulli'``, ``'system'``, ``'reservoir'`` (default).

        **Complexity:** O(n) — full scan with reservoir / bernoulli sampling.
        """
        if method not in ("bernoulli", "system", "reservoir"):
            raise HybridFrameError(
                f"Unknown sample method {method!r}; "
                f"use 'bernoulli', 'system', or 'reservoir'."
            )
        if isinstance(n, float):
            if not 0 <= n <= 100:
                raise HybridFrameError("Percentage n must be between 0 and 100.")
            sample_clause = f"{n} PERCENT ({method})"
        else:
            if n < 0:
                raise HybridFrameError("Sample n must be non-negative.")
            sample_clause = f"{n} ROWS ({method})"
        rel = self._relation
        try:
            self._relation = self._conn.sql(
                f"SELECT * FROM rel USING SAMPLE {sample_clause}"
            )
        except duckdb.Error as e:
            raise HybridFrameError(f"sample failed: {e}") from e
        return self

    def head(self, n: int = 5) -> "HybridFrame":
        if self._df is None and self._relation is None:
            return HybridFrame.from_pandas(pd.DataFrame())
        if self._engine is Engine.PANDAS_DATAFRAME:
            if self._df is None:
                return HybridFrame.from_pandas(pd.DataFrame())
            return HybridFrame.from_pandas(self._df.head(n))
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            try:
                limited = self._relation.limit(n)
                return HybridFrame.from_pandas(limited.df(), copy=False)
            except duckdb.Error as e:
                raise HybridFrameError(f"head failed: {e}") from e
        return HybridFrame.from_pandas(pd.DataFrame())

    def tail(self, n: int = 5) -> "HybridFrame":
        if self._df is None and self._relation is None:
            return HybridFrame.from_pandas(pd.DataFrame())
        if self._engine is Engine.PANDAS_DATAFRAME:
            if self._df is None:
                return HybridFrame.from_pandas(pd.DataFrame())
            return HybridFrame.from_pandas(self._df.tail(n))
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            try:
                count = self._relation.shape[0]
                offset = max(0, count - n)
                with _temp_view(self._conn, self._relation, "_hf_tail") as v:
                    result = self._conn.sql(
                        f"SELECT * FROM {v} LIMIT {n} OFFSET {offset}"
                    )
                    return HybridFrame.from_pandas(result.df(), copy=False)
            except duckdb.Error as e:
                raise HybridFrameError(f"tail failed: {e}") from e
        return HybridFrame.from_pandas(pd.DataFrame())

    @_ensure_engine(Engine.DUCKDB_RELATION)
    def groupby_agg(
        self,
        by: Sequence[str],
        agg_dict: Dict[str, Any],
    ) -> "HybridFrame":
        if not by:
            raise HybridFrameError("groupby_agg 'by' list must not be empty.")
        if not agg_dict:
            raise HybridFrameError("groupby_agg 'agg_dict' must not be empty.")
        by_quoted = ", ".join(f'"{c}"' for c in by)
        agg_parts: List[str] = []
        for col, func in agg_dict.items():
            if isinstance(func, str):
                funcs = [func]
            else:
                funcs = list(func)
            for f in funcs:
                sql_func = _normalise_agg(f)
                alias = f"{col}_{f}"
                if f.strip().lower() == "nunique":
                    agg_parts.append(f'COUNT(DISTINCT("{col}")) AS "{alias}"')
                else:
                    agg_parts.append(f'{sql_func}("{col}") AS "{alias}"')
        agg_expr = ", ".join(agg_parts)
        rel = self._relation
        sql = f"SELECT {by_quoted}, {agg_expr} FROM rel GROUP BY {by_quoted}"
        try:
            self._relation = self._conn.sql(sql)
        except duckdb.Error as e:
            raise HybridFrameError(f"Group-by aggregation failed: {e}") from e
        return self

    @_ensure_engine(Engine.DUCKDB_RELATION)
    def join(
        self,
        other: "HybridFrame",
        on: Union[str, List[str]],
        how: str = "inner",
    ) -> "HybridFrame":
        """Join two HybridFrame streams inside DuckDB before materialisation.

        **Complexity:** O(left × right) — hash join, full scan of both sides.

        Notes
        -----
        Uses DuckDB's Python local-variable scope-resolution for aliased joins.
        Compatible with duckdb >= 0.8.
        """
        other._to_duckdb()
        if other._conn is not self._conn:
            self._ensure_both_duckdb(other)
        if isinstance(on, str):
            keys = [on]
        else:
            keys = list(on)
        left_rel = self._relation
        right_rel = other._relation
        cond = " AND ".join(f'l."{k}" = r."{k}"' for k in keys)
        try:
            self._relation = self._conn.sql(
                f"SELECT * FROM left_rel l {how.upper()} JOIN right_rel r ON {cond}"
            )
        except duckdb.Error as e:
            raise HybridFrameError(f"Join failed: {e}") from e
        return self

    def _ensure_both_duckdb(self, other: "HybridFrame") -> None:
        """Transition both frames to DuckDB on the same connection."""
        other._to_duckdb()
        if other._conn is not self._conn:
            try:
                self._materialise_safely(other._relation)
                arrow_table = other._relation.fetch_arrow_table()
            except duckdb.Error as e:
                raise HybridFrameError(f"Cross-connection transfer failed: {e}") from e
            other._conn.close()
            other._conn = self._conn
            other._relation = self._conn.from_arrow(arrow_table)
            other._engine = Engine.DUCKDB_RELATION

    def union(self, other: "HybridFrame", all: bool = False) -> "HybridFrame":
        """Return the set union of two HybridFrames.

        Uses DuckDB SQL: ``SELECT * FROM left_rel UNION ALL SELECT * FROM right_rel``
        (or ``UNION`` when *all* is False).

        Parameters
        ----------
        other : HybridFrame
        all : bool
            If True, use ``UNION ALL`` (keep duplicates).  Default False.

        **Complexity:** O(n + m) — full scan of both sides.
        """
        self._to_duckdb()
        self._ensure_both_duckdb(other)
        rel = self._relation
        other_rel = other._relation
        set_op = "UNION ALL" if all else "UNION"
        try:
            self._relation = self._conn.sql(
                f"SELECT * FROM rel {set_op} SELECT * FROM other_rel"
            )
        except duckdb.Error as e:
            raise HybridFrameError(f"union failed: {e}") from e
        return self

    def intersect(self, other: "HybridFrame", all: bool = False) -> "HybridFrame":
        """Return the set intersection of two HybridFrames.

        Uses DuckDB SQL: ``SELECT * FROM left_rel INTERSECT ALL SELECT * FROM right_rel``
        (or ``INTERSECT`` when *all* is False).

        Parameters
        ----------
        other : HybridFrame
        all : bool
            If True, use ``INTERSECT ALL``.  Default False.

        **Complexity:** O(n + m) — full scan of both sides.
        """
        self._to_duckdb()
        self._ensure_both_duckdb(other)
        rel = self._relation
        other_rel = other._relation
        set_op = "INTERSECT ALL" if all else "INTERSECT"
        try:
            self._relation = self._conn.sql(
                f"SELECT * FROM rel {set_op} SELECT * FROM other_rel"
            )
        except duckdb.Error as e:
            raise HybridFrameError(f"intersect failed: {e}") from e
        return self

    def except_(self, other: "HybridFrame", all: bool = False) -> "HybridFrame":
        """Return the set difference of two HybridFrames.

        Uses DuckDB SQL: ``SELECT * FROM left_rel EXCEPT ALL SELECT * FROM right_rel``
        (or ``EXCEPT`` when *all* is False).

        Parameters
        ----------
        other : HybridFrame
        all : bool
            If True, use ``EXCEPT ALL``.  Default False.

        **Complexity:** O(n + m) — full scan of both sides.
        """
        self._to_duckdb()
        self._ensure_both_duckdb(other)
        rel = self._relation
        other_rel = other._relation
        set_op = "EXCEPT ALL" if all else "EXCEPT"
        try:
            self._relation = self._conn.sql(
                f"SELECT * FROM rel {set_op} SELECT * FROM other_rel"
            )
        except duckdb.Error as e:
            raise HybridFrameError(f"except_ failed: {e}") from e
        return self

    @_ensure_engine(Engine.DUCKDB_RELATION)
    def sql(self, query: str) -> "HybridFrame":
        with _temp_view(self._conn, self._relation, "_hf_sql") as view:
            resolved = query.replace("self", view)
            try:
                result = self._conn.sql(resolved)
                self._materialise_safely(result)
                pdf = result.df()
            except duckdb.Error as e:
                raise HybridFrameError(f"SQL query failed: {e}") from e
        self._df = pdf
        self._engine = Engine.PANDAS_DATAFRAME
        self._relation = None
        return self

    def show_plan(self) -> str:
        """Return the DuckDB query plan (EXPLAIN output) for debugging.

        **Complexity:** O(1) — generates the plan without executing the query.

        Returns
        -------
        str
            Formatted query plan from the DuckDB optimiser.
        """
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            return self._relation.explain()
        raise HybridFrameError("show_plan() requires a DuckDB relation. Call ._to_duckdb() first.")

    # ------------------------------------------------------------------
    # Pandas-engine methods  (feature engineering / reshaping)
    # ------------------------------------------------------------------
    @_ensure_engine(Engine.PANDAS_DATAFRAME)
    def reshape(self, method: str, **kwargs: Any) -> "HybridFrame":
        method_map: Dict[str, Callable] = {
            "pivot": lambda df, **kw: df.pivot(**kw),
            "melt": lambda df, **kw: df.melt(**kw),
            "explode": lambda df, **kw: df.explode(**kw),
            "pivot_table": lambda df, **kw: df.pivot_table(**kw),
        }
        fn = method_map.get(method)
        if fn is None:
            raise HybridFrameError(
                f"Unknown reshape method {method!r}. "
                f"Supported: {list(method_map)}"
            )
        result = fn(self._df, **kwargs)
        if isinstance(result, pd.DataFrame):
            self._df = result
        elif isinstance(result, pd.Series):
            self._df = result.to_frame(name=kwargs.get("values", "value"))
        else:
            self._df = pd.DataFrame(result)
        return self

    def time_series_impute(
        self,
        method: str = "ffill",
        datetime_col: Optional[str] = None,
    ) -> "HybridFrame":
        """Forward / backward fill nulls using LAST_VALUE window function.

        **Complexity:** O(n log n) — requires sort on datetime_col for DuckDB path.
        """
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if method not in ("ffill", "bfill"):
                raise HybridFrameError(
                    f"Unknown impute method {method!r}; use 'ffill' or 'bfill'."
                )
            order_col = f'"{datetime_col}"' if datetime_col else None
            if order_col:
                order_expr = f"ORDER BY {order_col}" + (" DESC" if method == "bfill" else "")
            else:
                order_expr = ""
            rel = self._relation
            exprs = []
            for c in self.columns:
                exprs.append(
                    f'COALESCE("{c}", LAST_VALUE("{c}" IGNORE NULLS) '
                    f'OVER ({order_expr} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) '
                    f'AS "{c}"'
                )
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"time_series_impute failed: {e}") from e
            return self
        self._to_pandas()
        if datetime_col is not None:
            self._df = self._df.sort_values(datetime_col)
        if method == "ffill":
            self._df = self._df.ffill()
        elif method == "bfill":
            self._df = self._df.bfill()
        else:
            raise HybridFrameError(
                f"Unknown impute method {method!r}; use 'ffill' or 'bfill'."
            )
        return self

    def one_hot_encode(
        self,
        columns: Sequence[str],
        drop_first: bool = False,
        dtype: type = np.int64,
    ) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if isinstance(columns, str):
                columns = [columns]
            rel = self._relation
            values: Dict[str, List[Any]] = {}
            for col in columns:
                try:
                    distinct = self._conn.sql(
                        f'SELECT DISTINCT {_sql_identifier(col)} FROM rel ORDER BY {_sql_identifier(col)}'
                    )
                    self._materialise_safely(distinct)
                    pdf = distinct.df()
                except duckdb.Error as e:
                    raise HybridFrameError(f"one_hot_encode failed: {e}") from e
                vals = pdf.iloc[:, 0].tolist()
                if drop_first and len(vals) > 1:
                    vals = vals[1:]
                values[col] = vals
            ddl_cast = "::INTEGER" if dtype == np.int64 else ""
            exprs = []
            for c in self.columns:
                if c in values:
                    for v in values[c]:
                        safe_val = _sql_literal(v)
                        col_name = str(v).replace("'", "").replace('"', "").replace(" ", "_")
                        exprs.append(
                            f'CASE WHEN "{c}" = {safe_val} THEN 1{ddl_cast} '
                            f'ELSE 0{ddl_cast} END AS "{c}_{col_name}"'
                        )
                else:
                    exprs.append(f'"{c}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"one_hot_encode failed: {e}") from e
            return self
        self._to_pandas()
        self._df = pd.get_dummies(
            self._df,
            columns=list(columns),
            drop_first=drop_first,
            dtype=dtype,
        )
        return self

    @_ensure_engine(Engine.PANDAS_DATAFRAME)
    def apply_row_logic(self, func: Callable, **kwargs: Any) -> "HybridFrame":
        """Apply a row-wise Python function (materialises to Pandas first).

        **Complexity:** O(n × func) — full materialisation + Python loop.
        """
        result = self._df.apply(func, axis=1, **kwargs)
        if isinstance(result, pd.DataFrame):
            self._df = result
        elif isinstance(result, pd.Series):
            if result.name is None:
                result.name = "apply_result"
            self._df[result.name] = result
        else:
            self._df["apply_result"] = result
        return self

    def rename(self, columns: Dict[str, str]) -> "HybridFrame":
        """Rename columns via alias projection.

        **Complexity:** O(1) (projection pushdown — rows are not scanned).
        """
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            existing = set(self.columns)
            for old, new in columns.items():
                if new in existing and new != old:
                    raise HybridFrameError(
                        f"Column {new!r} already exists. Cannot rename {old!r} to {new!r}."
                    )
            rel = self._relation
            exprs = []
            for c in self.columns:
                alias = columns.get(c, c)
                exprs.append(f'"{c}" AS "{alias}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"rename failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.rename(columns=columns)
        return self

    def drop(self, columns: Union[str, List[str]]) -> "HybridFrame":
        """Drop column(s) via projection.

        **Complexity:** O(1) (projection pushdown).
        """
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if isinstance(columns, str):
                columns = [columns]
            keep = [c for c in self.columns if c not in columns]
            if not keep:
                raise HybridFrameError("Cannot drop all columns.")
            return self.select(keep)
        self._to_pandas()
        if isinstance(columns, str):
            columns = [columns]
        existing = [c for c in columns if c in self._df.columns]
        if not existing:
            return self
        keep = [c for c in self._df.columns if c not in existing]
        if not keep:
            raise HybridFrameError("Cannot drop all columns.")
        self._df = self._df.drop(columns=existing)
        return self

    def fillna(self, value: Any) -> "HybridFrame":
        """Fill null values with a scalar or dict of per-column fill values.

        **Complexity:** O(n) — full scan with COALESCE per column.
        """
        if value is None:
            return self
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            is_dict = isinstance(value, dict)
            exprs = []
            for c in self.columns:
                if is_dict and c in value:
                    val = value[c]
                elif is_dict:
                    val = None
                else:
                    val = value
                if val is not None:
                    exprs.append(f'COALESCE("{c}", {_sql_literal(val)}) AS "{c}"')
                else:
                    exprs.append(f'"{c}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"fillna failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.fillna(value)
        return self

    def clip(
        self,
        lower: Union[float, Dict[str, float], None] = None,
        upper: Union[float, Dict[str, float], None] = None,
    ) -> "HybridFrame":
        """Clip values to a threshold range.

        DuckDB path uses ``LEAST/GREATEST`` per numeric column.
        Pandas path delegates to ``pd.DataFrame.clip``.

        Parameters
        ----------
        lower : float or dict, optional
            Minimum value (or per-column dict).
        upper : float or dict, optional
            Maximum value (or per-column dict).

        **Complexity:** O(n) — full scan.
        """
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if lower is None and upper is None:
                return self
            rel = self._relation
            num = set(self._get_numeric_columns())
            exprs = []
            for c in self.columns:
                lo = lower.get(c) if isinstance(lower, dict) else lower
                hi = upper.get(c) if isinstance(upper, dict) else upper
                if (lo is not None or hi is not None) and c in num:
                    if lo is not None and hi is not None:
                        exprs.append(f'LEAST(GREATEST("{c}", {_sql_literal(lo)}), {_sql_literal(hi)}) AS "{c}"')
                    elif lo is not None:
                        exprs.append(f'GREATEST("{c}", {_sql_literal(lo)}) AS "{c}"')
                    else:
                        exprs.append(f'LEAST("{c}", {_sql_literal(hi)}) AS "{c}"')
                else:
                    exprs.append(f'"{c}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"clip failed: {e}") from e
            return self
        self._to_pandas()
        num_cols = self._get_numeric_columns()
        if num_cols:
            with pd.option_context("mode.chained_assignment", None):
                for c in num_cols:
                    lo = lower.get(c) if isinstance(lower, dict) else lower
                    hi = upper.get(c) if isinstance(upper, dict) else upper
                    if lo is not None or hi is not None:
                        self._df[c] = self._df[c].clip(lower=lo, upper=hi)
        return self

    def astype(self, dtype: Union[str, Dict[str, str]]) -> "HybridFrame":
        """Cast columns to a new type.

        DuckDB path: ``SELECT col::new_type AS col`` (per column or dict of columns).
        Pandas path delegates to ``pd.DataFrame.astype``.

        Parameters
        ----------
        dtype : str or dict
            Target type (e.g. ``'BIGINT'``) or per-column dict like
            ``{'age': 'BIGINT', 'salary': 'DOUBLE'}``.

        **Complexity:** O(n) — full scan when materialised.
        """
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            is_dict = isinstance(dtype, dict)
            exprs = []
            for c in self.columns:
                target = dtype.get(c) if is_dict else dtype
                if target is not None:
                    exprs.append(f'CAST("{c}" AS {target}) AS "{c}"')
                else:
                    exprs.append(f'"{c}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"astype failed: {e}") from e
            return self
        self._to_pandas()
        if isinstance(dtype, str):
            pd_dtype = _DUCKDB_TO_NUMPY_TYPE.get(dtype.upper(), dtype)
        elif isinstance(dtype, dict):
            pd_dtype = {k: _DUCKDB_TO_NUMPY_TYPE.get(v.upper(), v) for k, v in dtype.items()}
        else:
            pd_dtype = dtype
        try:
            self._df = self._df.astype(pd_dtype)
        except Exception as e:
            raise HybridFrameError(f"astype failed: {e}") from e
        return self

    def isna(self) -> pd.DataFrame:
        if self._df is None and self._relation is None:
            return pd.DataFrame()
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            exprs = [f'{_sql_identifier(c)} IS NULL AS {_sql_identifier(c)}' for c in self.columns]
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                result = self._conn.sql(sql)
                self._materialise_safely(result)
                return result.df()
            except duckdb.Error as e:
                raise HybridFrameError(f"isna failed: {e}") from e
        self._to_pandas()
        return self._df.isna()

    def nunique(self) -> pd.Series:
        if self._df is None and self._relation is None:
            return pd.Series(dtype=int)
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            exprs = [f'COUNT(DISTINCT {_sql_identifier(c)}) AS {_sql_identifier(c)}' for c in self.columns]
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                result = self._conn.sql(sql)
                self._materialise_safely(result)
                return result.df().iloc[0]
            except duckdb.Error as e:
                raise HybridFrameError(f"nunique failed: {e}") from e
        self._to_pandas()
        return self._df.nunique()

    def value_counts(self, column: str) -> pd.Series:
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            sql = f'SELECT {_sql_identifier(column)}, COUNT(*) AS cnt FROM rel GROUP BY {_sql_identifier(column)} ORDER BY cnt DESC'
            try:
                result = self._conn.sql(sql)
                self._materialise_safely(result)
                pdf = result.df()
            except duckdb.Error as e:
                raise HybridFrameError(f"value_counts failed: {e}") from e
            return pdf.set_index(column)["cnt"]
        self._to_pandas()
        return self._df[column].value_counts()

    def dropna(self, axis: int = 0, how: str = "any", subset: Optional[Sequence[str]] = None) -> "HybridFrame":
        if how not in ("any", "all"):
            raise HybridFrameError(f"Unknown how={how!r}; use 'any' or 'all'.")
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            cols = subset if subset is not None else self.columns
            if isinstance(cols, str):
                cols = [cols]
            if how == "any":
                cond = " OR ".join(f'{_sql_identifier(c)} IS NULL' for c in cols)
            else:
                cond = " AND ".join(f'{_sql_identifier(c)} IS NULL' for c in cols)
            sql = f"SELECT * FROM rel WHERE NOT ({cond})"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"dropna failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.dropna(axis=axis, how=how, subset=subset)
        return self

    # ------------------------------------------------------------------
    # I/O: write to disk
    # ------------------------------------------------------------------
    def write_csv(self, path: Union[str, Path], **kwargs: Any) -> None:
        if self._df is None and self._relation is None:
            pd.DataFrame().to_csv(path, index=False, **kwargs)
            return
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None and not kwargs:
            self._relation.write_csv(str(path))
            return
        self.to_pandas().to_csv(path, index=False, **kwargs)

    def write_parquet(self, path: Union[str, Path], **kwargs: Any) -> None:
        if self._df is None and self._relation is None:
            pd.DataFrame().to_parquet(path, index=False, **kwargs)
            return
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None and not kwargs:
            self._relation.write_parquet(str(path))
            return
        self.to_pandas().to_parquet(path, index=False, **kwargs)

    # ------------------------------------------------------------------
    # Exploration & ML export
    # ------------------------------------------------------------------
    def describe(self) -> pd.DataFrame:
        if self._engine is Engine.DUCKDB_RELATION:
            if self._relation is None:
                return pd.DataFrame()
            with _temp_view(self._conn, self._relation, "_hf_desc") as view:
                try:
                    result = self._conn.sql(f"SUMMARIZE {view}")
                    self._materialise_safely(result)
                    return result.df()
                except duckdb.Error as e:
                    raise HybridFrameError(f"Describe failed: {e}") from e
        if self._df is not None:
            return self._df.describe()
        return pd.DataFrame()

    def to_pandas(self, force: bool = False, copy: bool = True) -> pd.DataFrame:
        if self._df is None and self._relation is None:
            return pd.DataFrame()
        if self._engine is Engine.PANDAS_DATAFRAME:
            if self._df is None:
                return pd.DataFrame()
            return self._df.copy() if copy else self._df
        self._to_pandas(force=force)
        return self._df.copy() if copy else self._df

    def to_pandas_iter(self, batch_size: int = 8192):
        for chunk_hf in self.fetch_chunked(batch_size=batch_size):
            yield chunk_hf.to_pandas(copy=False)

    def to_ml_ready(
        self,
        target_column: str,
        force: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return ``(X, y)`` tuple ready for Scikit‑Learn / PyTorch / XGBoost.

        Parameters
        ----------
        target_column : str
            Column name to use as the label ``y``.
        force : bool
            Bypass OOM guard during materialisation.

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        df = self.to_pandas(force=force)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info("ML-ready split: X %s, y %s", X.shape, y.shape)
        return X, y

    # ------------------------------------------------------------------
    # Streaming / chunked iteration
    # ------------------------------------------------------------------
    def fetch_chunked(self, batch_size: int = 8192, use_arrow: bool = False):
        if self._df is None and self._relation is None:
            return
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            if use_arrow and HAS_PYARROW:
                reader = self._relation.fetch_arrow_reader(batch_size)
                for batch in reader:
                    if len(batch) == 0:
                        break
                    yield pa.Table.from_batches([batch])
            else:
                with _temp_view(self._conn, self._relation, "_hf_chunk") as v:
                    result = self._conn.execute(f"SELECT * FROM {v}")
                    while True:
                        with MemoryTracker("fetch_chunked", f"batch={batch_size}"):
                            chunk = result.fetch_df_chunk(batch_size)
                        if len(chunk) == 0:
                            break
                        chunk_est = len(chunk) * len(self.columns) * 50 / (1024 ** 3)
                        if chunk_est > 0.1:
                            avail = _available_ram_gb()
                            if chunk_est > avail:
                                logger.warning(
                                    "Chunk memory ~%.2f GB exceeds available %.1f GB. "
                                    "Reduce batch_size.",
                                    chunk_est,
                                    avail,
                                )
                        yield HybridFrame.from_pandas(chunk, copy=False)
        else:
            df = self._df if self._engine is Engine.PANDAS_DATAFRAME else self.to_pandas(copy=False)
            if use_arrow and HAS_PYARROW:
                for start in range(0, len(df), batch_size):
                    chunk = df.iloc[start:start + batch_size]
                    yield pa.Table.from_pandas(chunk)
            else:
                for start in range(0, len(df), batch_size):
                    yield HybridFrame.from_pandas(df.iloc[start:start + batch_size], copy=False)

    def to_arrow_reader(self, batch_size: int = 8192):
        """Return a ``pyarrow.RecordBatchReader`` for zero-copy streaming.

        Parameters
        ----------
        batch_size : int
            Number of rows per Arrow record batch (default 8192).

        Returns
        -------
        pyarrow.RecordBatchReader
        """
        if not HAS_PYARROW:
            raise HybridFrameError("pyarrow is required for to_arrow_reader(). Install with: pip install pyarrow")
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            return self._relation.fetch_arrow_reader(batch_size)
        df = self._df if self._engine is Engine.PANDAS_DATAFRAME else self.to_pandas(copy=False)
        schema = pa.Schema.from_pandas(df)

        def _gen():
            for start in range(0, len(df), batch_size):
                yield pa.RecordBatch.from_pandas(df.iloc[start:start + batch_size])
        return pa.RecordBatchReader.from_batches(schema, _gen())

    def to_torch_dataloader(
        self,
        target_column: str,
        batch_size: int = 32,
        shuffle: bool = False,
        **kwargs: Any,
    ):
        """Return a PyTorch ``DataLoader`` that streams batches from disk.

        Each yielded element is a ``(X, y)`` tuple of tensors where
        ``y`` is the column named by *target_column* and ``X`` is every
        other column.

        Parameters
        ----------
        target_column : str
            Column to use as the label ``y``.
        batch_size : int
            Rows per batch (default 32).
        shuffle : bool
            Ignored (data is always sequential over an immutable relation).
        **kwargs
            Extra arguments forwarded to ``torch.utils.data.DataLoader``.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        if not HAS_TORCH:
            raise HybridFrameError(
                "torch is required for to_torch_dataloader(). Install with: pip install torch"
            )

        class _HybridIterableDataset(IterableDataset):
            def __init__(self, hf: "HybridFrame", target: str, bs: int):
                super().__init__()
                self._hf = hf
                self._target = target
                self._bs = bs

            def __iter__(self):
                for chunk in self._hf.fetch_chunked(batch_size=self._bs):
                    pdf = chunk.to_pandas(copy=False)
                    X = pdf.drop(columns=[self._target]).values
                    y = pdf[self._target].values
                    yield torch.from_numpy(X), torch.from_numpy(y)

        dataset = _HybridIterableDataset(self, target_column, batch_size)
        return DataLoader(dataset, batch_size=None, **kwargs)

    def memory_usage(self, deep: bool = False) -> pd.Series:
        if self._engine is Engine.PANDAS_DATAFRAME and self._df is not None:
            return self._df.memory_usage(deep=deep)
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            est = _estimate_relation_memory(self._relation)
            total_bytes = int(est * (1024 ** 3))
            n = len(self.columns)
            if n == 0:
                return pd.Series(dtype=int)
            per_col = total_bytes // n
            return pd.Series(
                {c: per_col for c in self.columns},
                name="memory_usage",
            )
        return pd.Series(dtype=int)


    # ------------------------------------------------------------------
    # Numeric column detection helper
    # ------------------------------------------------------------------
    def _get_numeric_columns(self) -> List[str]:
        dt = self.dtypes
        return [
            c for c in dt.index
            if "int" in str(dt[c]).lower() or "float" in str(dt[c]).lower()
            or "double" in str(dt[c]).lower() or "decimal" in str(dt[c]).lower()
        ]

    # ------------------------------------------------------------------
    # isnull / notnull
    # ------------------------------------------------------------------
    def isnull(self) -> pd.DataFrame:
        return self.isna()

    def notnull(self) -> pd.DataFrame:
        return ~self.isna()

    # ------------------------------------------------------------------
    # replace
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_key_compatible(mapping_keys: List[Any], dtype: Any) -> bool:
        s = str(dtype).lower()
        is_num = any(t in s for t in ["int", "float", "double", "decimal", "numeric"])
        is_str = any(t in s for t in ["varchar", "text", "string"]) or s == "object"
        is_bool = "bool" in s
        for k in mapping_keys:
            if isinstance(k, bool) and is_bool:
                return True
            if isinstance(k, str) and is_str:
                return True
            if isinstance(k, (int, float, np.integer, np.floating)) and is_num:
                return True
        return False

    def replace(self, to_replace: Union[Dict[Any, Any], Dict[str, Dict[Any, Any]]]) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            is_per_col = to_replace and all(isinstance(v, dict) for v in to_replace.values())
            exprs = []
            for c in self.columns:
                mapping = to_replace.get(c, {}) if is_per_col else to_replace
                if mapping and self._replace_key_compatible(list(mapping.keys()), self.dtypes[c]):
                    parts = " ".join(
                        f'WHEN "{c}" = {_sql_literal(k)} THEN {_sql_literal(v)}'
                        for k, v in mapping.items()
                    )
                    exprs.append(f"CASE {parts} ELSE \"{c}\" END AS \"{c}\"")
                else:
                    exprs.append(f'"{c}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"replace failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.replace(to_replace)
        return self

    # ------------------------------------------------------------------
    # where
    # ------------------------------------------------------------------
    def where(self, cond: str, other: Any = None) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            is_dict = isinstance(other, dict)
            num = set(self._get_numeric_columns())
            exprs = []
            for c in self.columns:
                val = other.get(c, None) if is_dict else other
                if val is not None and c in num:
                    exprs.append(
                        f'CASE WHEN ({cond}) THEN "{c}" ELSE {_sql_literal(val)} END AS "{c}"'
                    )
                elif val is None and c in num:
                    exprs.append(
                        f'CASE WHEN ({cond}) THEN "{c}" ELSE NULL END AS "{c}"'
                    )
                else:
                    exprs.append(f'"{c}"')
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"where failed: {e}") from e
            return self
        self._to_duckdb()
        return self.where(cond, other=other)

    # ------------------------------------------------------------------
    # between
    # ------------------------------------------------------------------
    def between(self, column: str, low: Any, high: Any) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            sql = (
                f'SELECT * FROM rel WHERE "{column}" '
                f"BETWEEN {_sql_literal(low)} AND {_sql_literal(high)}"
            )
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"between failed: {e}") from e
            return self
        self._to_pandas()
        try:
            self._df = self._df[self._df[column].between(low, high)]
        except KeyError as e:
            raise HybridFrameError(f"Column not found: {e}") from e
        return self

    # ------------------------------------------------------------------
    # idxmin / idxmax
    # ------------------------------------------------------------------
    def idxmin(self, column: str) -> int:
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            with _temp_view(self._conn, self._relation, "_hf_idx") as v:
                sql = (
                    f'SELECT _hf_rn - 1 AS idx FROM ('
                    f'SELECT *, ROW_NUMBER() OVER () AS _hf_rn FROM {v}'
                    f') sub ORDER BY {_sql_identifier(column)} ASC LIMIT 1'
                )
                try:
                    result = self._conn.sql(sql)
                    self._materialise_safely(result)
                    return int(result.df().iloc[0, 0])
                except duckdb.Error as e:
                    raise HybridFrameError(f"idxmin failed: {e}") from e
        self._to_pandas()
        try:
            return int(self._df[column].to_numpy().argmin())
        except Exception as e:
            raise HybridFrameError(f"idxmin failed: {e}") from e

    def idxmax(self, column: str) -> int:
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            with _temp_view(self._conn, self._relation, "_hf_idx") as v:
                sql = (
                    f'SELECT _hf_rn - 1 AS idx FROM ('
                    f'SELECT *, ROW_NUMBER() OVER () AS _hf_rn FROM {v}'
                    f') sub ORDER BY {_sql_identifier(column)} DESC LIMIT 1'
                )
                try:
                    result = self._conn.sql(sql)
                    self._materialise_safely(result)
                    return int(result.df().iloc[0, 0])
                except duckdb.Error as e:
                    raise HybridFrameError(f"idxmax failed: {e}") from e
        self._to_pandas()
        try:
            return int(self._df[column].to_numpy().argmax())
        except Exception as e:
            raise HybridFrameError(f"idxmax failed: {e}") from e

    # ------------------------------------------------------------------
    # abs / round
    # ------------------------------------------------------------------
    def abs(self) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            num = set(self._get_numeric_columns())
            exprs = [
                f'ABS("{c}") AS "{c}"' if c in num else f'"{c}"'
                for c in self.columns
            ]
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"abs failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.abs()
        return self

    def round(self, decimals: int = 0) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            num = set(self._get_numeric_columns())
            exprs = [
                f'ROUND("{c}", {decimals}) AS "{c}"'
                if c in num else f'"{c}"'
                for c in self.columns
            ]
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"round failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.round(decimals)
        return self

    # ------------------------------------------------------------------
    # diff
    # ------------------------------------------------------------------
    def diff(self, order_by: Optional[str] = None) -> "HybridFrame":
        if self._df is None and self._relation is None:
            return HybridFrame()
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            num = set(self._get_numeric_columns())
            if order_by:
                order_expr = f'ORDER BY "{order_by}"'
            else:
                with _temp_view(self._conn, rel, "_hf_diff") as v:
                    result = self._conn.sql(
                        f"SELECT COUNT(*) AS cnt FROM {v}"
                    ).df()
                    if result.iloc[0, 0] == 0:
                        return self
                order_expr = "ORDER BY (SELECT NULL)"
            exprs = [
                f'("{c}" - LAG("{c}") OVER ({order_expr})) AS "{c}"'
                if c in num else f'"{c}"'
                for c in self.columns
            ]
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"diff failed: {e}") from e
            return self
        self._to_pandas()
        self._df = self._df.diff()
        return self

    # ------------------------------------------------------------------
    # Cumulative operations
    # ------------------------------------------------------------------
    def cumsum(self, order_by: Optional[str] = None) -> "HybridFrame":
        return self._cum_op("SUM", order_by)

    def cumprod(self, order_by: Optional[str] = None) -> "HybridFrame":
        return self._cum_op("PRODUCT", order_by)

    def cummin(self, order_by: Optional[str] = None) -> "HybridFrame":
        return self._cum_op("MIN", order_by)

    def cummax(self, order_by: Optional[str] = None) -> "HybridFrame":
        return self._cum_op("MAX", order_by)

    def _cum_op(self, op: str, order_by: Optional[str] = None) -> "HybridFrame":
        if self._engine is Engine.DUCKDB_RELATION and self._relation is not None:
            rel = self._relation
            num = set(self._get_numeric_columns())
            if order_by:
                order_expr = f'ORDER BY "{order_by}"'
            else:
                with _temp_view(self._conn, rel, "_hf_cum") as v:
                    result = self._conn.sql(
                        f"SELECT COUNT(*) AS cnt FROM {v}"
                    ).df()
                    if result.iloc[0, 0] == 0:
                        return self
                order_expr = "ORDER BY (SELECT NULL)"
            window = f"OVER ({order_expr} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
            exprs = [
                f'{op}("{c}") {window} AS "{c}"'
                if c in num else f'"{c}"'
                for c in self.columns
            ]
            sql = f"SELECT {', '.join(exprs)} FROM rel"
            try:
                self._relation = self._conn.sql(sql)
            except duckdb.Error as e:
                raise HybridFrameError(f"{op.lower()} failed: {e}") from e
            return self
        self._to_pandas()
        pdf = self._df
        if op == "SUM":
            self._df = pdf.cumsum()
        elif op == "PRODUCT":
            self._df = pdf.cumprod()
        elif op == "MIN":
            self._df = pdf.cummin()
        elif op == "MAX":
            self._df = pdf.cummax()
        return self


# ===================================================================
# Internal helpers (module-level)
# ===================================================================


def _normalise_agg(func: str) -> str:
    mapping = {
        "avg": "AVG",
        "mean": "AVG",
        "sum": "SUM",
        "count": "COUNT",
        "min": "MIN",
        "max": "MAX",
        "std": "STDDEV_SAMP",
        "var": "VAR_SAMP",
        "nunique": "COUNT(DISTINCT",
        "median": "MEDIAN",
        "first": "FIRST",
        "last": "LAST",
    }
    norm = func.strip().lower()
    mapped = mapping.get(norm)
    if mapped is None:
        return func.upper()
    return mapped


_ALLOWED_CSV_KWARGS = frozenset({
    "delimiter", "sep", "header", "names", "skiprows", "null_value",
    "escapechar", "quotechar", "dateformat", "timestampformat",
    "sample_size", "all_varchar", "normalize_names", "encoding",
    "compression", "parallel", "skip", "auto_detect",
})


def _validate_csv_kwargs(**kwargs: Any) -> None:
    bad = [k for k in kwargs if k not in _ALLOWED_CSV_KWARGS]
    if bad:
        raise HybridFrameError(
            f"Unrecognized CSV kwargs: {bad}. "
            f"Allowed: {sorted(_ALLOWED_CSV_KWARGS)}"
        )


_ALLOWED_PQ_KWARGS = frozenset({
    "compression", "row_group_size", "binary_as_string",
    "file_row_number", "hive_partitioning", "union_by_name",
})


def _validate_pq_kwargs(**kwargs: Any) -> None:
    bad = [k for k in kwargs if k not in _ALLOWED_PQ_KWARGS]
    if bad:
        raise HybridFrameError(
            f"Unrecognized Parquet kwargs: {bad}. "
            f"Allowed: {sorted(_ALLOWED_PQ_KWARGS)}"
        )


def _csv_kwargs_to_sql(**kwargs: Any) -> str:
    if not kwargs:
        return ""
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            parts.append(f"{k}={str(v).lower()}")
        elif isinstance(v, str):
            parts.append(f"{k}='{v.replace(chr(39), chr(39)+chr(39))}'")
        else:
            parts.append(f"{k}={v}")
    return ", " + ", ".join(parts)


def _pq_kwargs_to_sql(**kwargs: Any) -> str:
    return _csv_kwargs_to_sql(**kwargs)


# ===================================================================
# Demo:  End-to-end ML pipeline
# ===================================================================
if __name__ == "__main__":
    import tempfile

    _SEED = 42
    _N_ROWS = 50_000

    np.random.seed(_SEED)

    print("=" * 60)
    print("HybridFrame Demo — End-to-End ML Pipeline")
    print("=" * 60)

    print("\n[1] Generating synthetic customer dataset (%s rows) ..." % _N_ROWS)
    demo_df = pd.DataFrame(
        {
            "customer_id": range(_N_ROWS),
            "age": np.random.randint(18, 75, _N_ROWS),
            "income": np.random.lognormal(mean=10.5, sigma=0.8, size=_N_ROWS).astype(
                np.float32
            ),
            "education": np.random.choice(
                ["High School", "Bachelor", "Master", "PhD"], _N_ROWS,
                p=[0.3, 0.4, 0.2, 0.1],
            ),
            "city": np.random.choice(
                ["New York", "London", "Tokyo", "Berlin", "Bangalore"], _N_ROWS,
                p=[0.3, 0.25, 0.2, 0.15, 0.1],
            ),
            "purchase_amount": np.random.gamma(shape=5, scale=20, size=_N_ROWS).astype(
                np.float32
            ),
            "purchase_date": pd.date_range(
                "2023-01-01", periods=_N_ROWS, freq="h"
            ) + pd.to_timedelta(np.random.randint(0, 3600, _N_ROWS), unit="s"),
            "category": np.random.choice(
                ["Electronics", "Clothing", "Food", "Books", "Sports"], _N_ROWS,
                p=[0.25, 0.3, 0.2, 0.15, 0.1],
            ),
            "rating": np.random.randint(1, 6, _N_ROWS).astype(np.int8),
        }
    )

    demo_df.loc[::7, "purchase_amount"] = np.nan

    csv_path = Path(tempfile.gettempdir()) / "hybridframe_demo.csv"
    demo_df.to_csv(csv_path, index=False)
    print(f"    Saved to {csv_path}")

    print("\n[2] Loading CSV lazily via DuckDB ...")
    hf_core = HybridFrame.from_csv(csv_path)
    print(f"    {hf_core}")

    print("\n[3] Out-of-core filtering & projection (50K -> 29.7K) ...")
    hf_core = hf_core.filter("income > 30000")
    hf_core = hf_core.select(["customer_id", "age", "income", "education", "city",
                              "purchase_amount", "purchase_date", "category", "rating"])
    print(f"    After filter  → {hf_core}")

    hf_core_df = hf_core.to_pandas(copy=False)

    print("[4a] Group-by aggregation (DuckDB SQL) ...")
    hf_agg = HybridFrame.from_pandas(hf_core_df, copy=False).groupby_agg(
        by=["category", "city"],
        agg_dict={"purchase_amount": "sum", "income": "avg"},
    )
    print(f"    Aggregated    → {hf_agg}")

    print("\n[5a] Reshape: pivot (auto → Pandas) ...")
    hf_pivot = hf_agg.reshape(
        "pivot_table",
        index="category",
        columns="city",
        values="purchase_amount_sum",
        aggfunc="first",
    )
    print(f"    Pivoted shape → {hf_pivot.shape}")

    print("\n[6b] Join two HybridFrames ...")
    hf_left = HybridFrame.from_pandas(
        pd.DataFrame({"city": ["New York", "London", "Tokyo"],
                       "avg_temp": [22, 15, 18]})
    )
    hf_joined = HybridFrame.from_pandas(hf_core_df, copy=False).groupby_agg(
        by=["city"], agg_dict={"purchase_amount": "sum"}
    ).join(hf_left, on="city", how="left")
    print(f"    Joined        → {hf_joined}")

    print("\n[7c] Impute missing purchase_amount (ffill by date, bfill fallback) ...")
    hf_impute = HybridFrame.from_pandas(hf_core_df, copy=False)
    hf_impute.time_series_impute(method="ffill", datetime_col="purchase_date")
    if hf_impute["purchase_amount"].isna().sum():
        hf_impute.time_series_impute(method="bfill", datetime_col="purchase_date")
    remaining_na = hf_impute["purchase_amount"].isna().sum()
    assert remaining_na == 0, "All NaN values should be imputed"
    print(f"    Remaining NaN → {remaining_na}")

    print("\n[8c] One-hot encoding for ML ...")
    hf_encoded = hf_impute.one_hot_encode(
        columns=["education", "city", "category"],
        drop_first=True,
    )
    print(f"    Encoded shape → {hf_encoded.shape}")

    print("\n[9c] Row-level feature engineering (using public .rename()) ...")
    hf_encoded = (
        hf_encoded
        .apply_row_logic(lambda row: "high" if row["income"] > 50000 else "standard")
        .rename({"apply_result": "income_tier"})
        .apply_row_logic(
            lambda row: row["purchase_amount"] / row["age"]
            if row["age"] > 0 else 0.0
        )
        .rename({"apply_result": "spend_per_year"})
    )
    print("    New columns: 'income_tier', 'spend_per_year'")

    print("\n[10c] Summary statistics ...")
    desc = hf_encoded.describe()
    print(f"    Describe output → shape {desc.shape}")

    print("\n[11c] Export to ML framework ...")
    X, y = hf_encoded.to_ml_ready(target_column="purchase_amount")
    print(f"    X: {X.shape}  |  y: {y.shape}")
    print(f"    Feature columns ({len(X.columns)}):")
    for col in X.columns[:10]:
        print(f"      • {col}")
    if len(X.columns) > 10:
        print(f"      … and {len(X.columns) - 10} more")

    print("\n[12c] Quick-fit Random Forest (proving ML readiness) ...")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        X_train, X_test, y_train, y_test = train_test_split(
            X.select_dtypes(include=[np.number]), y, test_size=0.2,
            random_state=_SEED,
        )
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        y_train = y_train.fillna(y_train.median())
        y_test = y_test.fillna(y_test.median())

        model = RandomForestRegressor(
            n_estimators=50, max_depth=8, n_jobs=-1, random_state=_SEED
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"    R² score      → {r2:.4f}")
    except ImportError:
        print("    scikit-learn not installed – skipping fit.")

    print("\n[13] New API methods demo ...")
    hf_sample = HybridFrame.from_pandas(hf_core_df, copy=False)
    print(f"    .columns      → {hf_sample.columns[:5]}...")
    print(f"    .shape        → {hf_sample.shape}")
    print(f"    .head(2)      → {hf_sample.head(2).shape}")
    print(f"    .tail(2)      → {hf_sample.tail(2).shape}")
    hf_sorted = hf_sample.sort_values("purchase_amount", ascending=False).limit(3)
    print(f"    sort+limit    → {hf_sorted.to_pandas()[['customer_id', 'purchase_amount']].values}")
    hf_sample.write_csv(Path(tempfile.gettempdir()) / "hf_export.csv")
    print(f"    write_csv     → exported")
    out = hf_sample.nunique()
    print(f"    .nunique()    → {dict(list(out.items())[:5])}")

    print(f"\n[14] fetch_chunked streaming demo ...")
    hf_big = HybridFrame.from_pandas(pd.DataFrame({"x": range(1000)}))
    total = 0
    for i, chunk in enumerate(hf_big.fetch_chunked(batch_size=300)):
        total += chunk.shape[0]
        if i < 3:
            print(f"    chunk {i}: shape={chunk.shape}")
    print(f"    streamed {total} rows across {i+1} chunks")

    for h in [hf_core, hf_agg, hf_pivot, hf_joined, hf_impute,
              hf_encoded, hf_left, hf_sample]:
        h.close()
    print("\n" + "=" * 60)
    print("Pipeline complete – HybridFrame demonstrated end-to-end.")
    print("=" * 60)
