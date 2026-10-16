from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .model import SPLIT_COL, TIME_COL, Stream

CACHE_VERSION = "3"


def cache_dir() -> Path:
    env = os.environ.get("TELEOP_ANALYSIS_CACHE")
    if env:
        d = Path(env)
    elif os.name == "nt":
        d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "teleop-analysis" / "cache"
    else:
        d = Path.home() / ".cache" / "teleop-analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(p: Path) -> str:
    try:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return f"{p}|missing"


def cache_path(stream: Stream) -> Path:
    loader = stream.loader
    parts = [CACHE_VERSION, type(loader).__name__, getattr(loader, "version", "1"),
             getattr(stream, "cache_id", None) or stream.info.get("cache_id", ""), _stamp(stream.path)]
    parts += [_stamp(d) for d in stream.deps]
    h = hashlib.sha1("||".join(parts).encode()).hexdigest()[:24]
    return cache_dir() / f"{stream.path.stem[:40]}_{h}.parquet"


class TableStore:
    def __init__(self, max_columns: int = 600):
        self._cols: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._files: dict[str, pq.ParquetFile] = {}
        self.max_columns = max_columns
        self.progress = None

    def ensure(self, stream: Stream) -> Path:
        out = cache_path(stream)
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            try:
                stream.loader.write(stream, tmp, self.progress, self)
                os.replace(tmp, out)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        return out

    def _pf(self, stream: Stream) -> pq.ParquetFile:
        out = self.ensure(stream)
        k = str(out)
        if k not in self._files:
            self._files[k] = pq.ParquetFile(out)
        return self._files[k]

    def available(self, stream: Stream) -> list[str]:
        return [n for n in self._pf(stream).schema_arrow.names]

    def _raw(self, stream: Stream, col: str) -> np.ndarray:
        out = cache_path(stream)
        k = (str(out), col)
        if k in self._cols:
            self._cols.move_to_end(k)
            return self._cols[k]
        pf = self._pf(stream)
        if col not in pf.schema_arrow.names:
            raise KeyError(col)
        arr = pf.read(columns=[col]).column(0)
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            v = np.asarray(arr.to_pylist(), dtype=object)
        elif col == TIME_COL:
            v = arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        else:
            v = arr.cast(pa.float64()).to_numpy(zero_copy_only=False)
        self._cols[k] = v
        while len(self._cols) > self.max_columns:
            self._cols.popitem(last=False)
        return v

    def _rows(self, stream: Stream) -> np.ndarray | None:
        if stream.split_value is None:
            return None
        k = (str(cache_path(stream)), SPLIT_COL, str(stream.split_value))
        if k in self._cols:
            return self._cols[k]
        s = self._raw(stream, SPLIT_COL)
        m = s == stream.split_value
        self._cols[k] = m
        return m

    def time_ns(self, stream: Stream) -> np.ndarray:
        t = self._raw(stream, TIME_COL)
        m = self._rows(stream)
        return t if m is None else t[m]

    def column(self, stream: Stream, col: str) -> np.ndarray:
        v = self._raw(stream, col)
        m = self._rows(stream)
        return v if m is None else v[m]

    def clear(self):
        self._cols.clear()
        self._files.clear()
