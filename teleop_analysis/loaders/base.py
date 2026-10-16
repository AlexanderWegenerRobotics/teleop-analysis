from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..model import CLOCK_AVATAR, CLOCK_INTERFACE, CLOCK_OPERATOR, SPLIT_COL, TIME_COL

INTERFACE_NAMES = {"command", "stream", "frames", "events"}
NS_TIME_PRIORITY = ["wall_clock_ns", "timestamp_ns", "time_ns", "t_ns", "stamp_ns", "capture_time_ns"]
S_TIME_PRIORITY = ["timestamp", "time_s", "time", "t", "sim_time", "sec", "seconds"]
SPLIT_CANDIDATES = ["device_index", "device_id", "arm_index", "arm_id"]
EPOCH_NS_MIN = 1e17


class Loader:
    version = "1"

    def match(self, path: Path) -> bool:
        return False

    def discover(self, root, path: Path, rel: str) -> list:
        return []

    def write(self, stream, out: Path, progress=None, store=None) -> None:
        table = self.build(stream, store)
        pq.write_table(table, out)

    def build(self, stream, store=None) -> pa.Table:
        raise NotImplementedError


def guess_clock(rel: str) -> str:
    parts = [p.lower() for p in Path(rel).parts]
    if any("operator" in p for p in parts[:-1]):
        return CLOCK_OPERATOR
    if any("interface" in p for p in parts[:-1]):
        return CLOCK_INTERFACE
    stem = Path(rel).name.split(".")[0].lower()
    if stem in INTERFACE_NAMES:
        return CLOCK_INTERFACE
    return CLOCK_AVATAR


def sniff_sep(line: str) -> str:
    counts = {s: line.count(s) for s in (";", ",", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_head(path: Path, nbytes: int = 256 * 1024) -> tuple[str, list[str]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        text = f.read(nbytes)
    lines = text.splitlines()
    if len(text) >= nbytes and lines:
        lines = lines[:-1]
    return text, lines


def sample_frame(path: Path, sep: str, nrows: int = 300) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, nrows=nrows, dtype=str, keep_default_na=False,
                           on_bad_lines="skip", engine="c")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def to_float(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


_INT_RE = re.compile(r"^\s*-?\d+\s*$")


def to_ns(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.iinfo(np.int64).min, dtype=np.int64)
    v = s.astype(str).str.strip()
    is_int = v.str.fullmatch(r"-?\d+").fillna(False).to_numpy()
    if is_int.any():
        out[is_int] = v[is_int].astype(np.int64).to_numpy()
    rest = ~is_int
    if rest.any():
        f = pd.to_numeric(v[rest], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        ok = np.isfinite(f)
        idx = np.flatnonzero(rest)
        out[idx[ok]] = f[ok].astype(np.int64)
    return out


NS_INVALID = np.iinfo(np.int64).min


def is_numeric_sample(values: pd.Series) -> bool:
    v = values[values.astype(str).str.strip() != ""]
    if len(v) == 0:
        return True
    f = pd.to_numeric(v, errors="coerce")
    return f.notna().mean() > 0.8


def looks_epoch_ns(values: pd.Series) -> bool:
    f = pd.to_numeric(values, errors="coerce").dropna()
    return len(f) > 0 and float(f.median()) > EPOCH_NS_MIN


def choose_time(sample: pd.DataFrame, prefer: list[str] | None = None) -> tuple[str | None, str]:
    cols = list(sample.columns)
    for c in (prefer or []):
        if c in cols:
            return c, ("ns" if c.endswith("_ns") else "s")
    for c in NS_TIME_PRIORITY:
        if c in cols and looks_epoch_ns(sample[c]):
            return c, "ns"
    for c in cols:
        if c.endswith("_ns") and looks_epoch_ns(sample[c]):
            return c, "ns"
    for c in S_TIME_PRIORITY:
        if c in cols:
            f = pd.to_numeric(sample[c], errors="coerce").dropna()
            if len(f) == 0:
                continue
            m = float(f.median())
            if m > 1e17:
                return c, "ns"
            if m > 1e14:
                return c, "us"
            if m > 1e11:
                return c, "ms"
            if m > 1e8:
                return c, "s_epoch"
            return c, "s"
    if len(sample) == 0:
        for c in NS_TIME_PRIORITY + S_TIME_PRIORITY:
            if c in cols:
                return c, ("ns" if c.endswith("_ns") else "s")
    return None, "row"


UNIT_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s_epoch": 1_000_000_000, "s": 1_000_000_000}


def time_to_ns(s: pd.Series, unit: str) -> np.ndarray:
    if unit == "ns":
        return to_ns(s)
    f = to_float(s) * UNIT_NS[unit]
    out = np.full(len(f), NS_INVALID, dtype=np.int64)
    ok = np.isfinite(f)
    out[ok] = np.round(f[ok]).astype(np.int64)
    return out


def table_from(t_ns: np.ndarray, cols: dict[str, np.ndarray], split: np.ndarray | None = None,
               strings: dict[str, list] | None = None) -> pa.Table:
    data = {TIME_COL: pa.array(t_ns, type=pa.int64())}
    for k, v in cols.items():
        data[k] = pa.array(np.asarray(v, dtype=np.float64), type=pa.float64())
    for k, v in (strings or {}).items():
        data[k] = pa.array([None if x is None else str(x) for x in v], type=pa.string())
    if split is not None:
        data[SPLIT_COL] = pa.array([str(x) for x in split], type=pa.string())
    return pa.table(data)


def empty_table(columns: list[str], with_split: bool = False) -> pa.Table:
    fields = [pa.field(TIME_COL, pa.int64())] + [pa.field(c, pa.float64()) for c in columns]
    if with_split:
        fields.append(pa.field(SPLIT_COL, pa.string()))
    return pa.schema(fields).empty_table()
