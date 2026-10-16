from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..model import CLOCK_RELATIVE, SIGNALS, SPLIT_COL, TIME_COL, Stream
from .base import (NS_INVALID, UNIT_NS, SPLIT_CANDIDATES, Loader, choose_time, guess_clock, is_numeric_sample,
                   looks_epoch_ns, read_head, sample_frame, sniff_sep, time_to_ns, to_float, to_ns)

CHUNK_ROWS = 250_000


class CsvSpec:
    def __init__(self, sep, time_col, unit, numeric, ns_rel, ns_dur, split_col):
        self.sep = sep
        self.time_col = time_col
        self.unit = unit
        self.numeric = numeric
        self.ns_rel = ns_rel
        self.ns_dur = ns_dur
        self.split_col = split_col

    def out_columns(self) -> list[str]:
        cols = list(self.numeric)
        cols += [rel_name(c) for c in self.ns_rel]
        cols += [dur_name(c) for c in self.ns_dur]
        return cols


def rel_name(c: str) -> str:
    return f"{c[:-3]} - t [ms]"


def dur_name(c: str) -> str:
    return f"{c[:-3]} [ms]"


class CsvLoader(Loader):
    version = "4"
    suffixes = (".csv", ".tsv")
    prefer_time: list[str] | None = None
    kind = SIGNALS

    def match(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    def analyse(self, path: Path) -> tuple[CsvSpec, pd.DataFrame]:
        text, lines = read_head(path)
        if not lines:
            raise EmptyFile()
        sep = sniff_sep(lines[0])
        sample = sample_frame(path, sep)
        if sample.columns.empty:
            raise EmptyFile()
        time_col, unit = choose_time(sample, self.prefer_time)
        split_col = None
        for c in SPLIT_CANDIDATES:
            if c in sample.columns:
                u = sample[c].astype(str).str.strip()
                u = u[u != ""].unique()
                if 2 <= len(u) <= 8:
                    split_col = c
                break
        numeric, ns_rel, ns_dur = [], [], []
        for c in sample.columns:
            if c == time_col or c == split_col or not str(c).strip():
                continue
            if not is_numeric_sample(sample[c]):
                continue
            if str(c).endswith("_ns"):
                if looks_epoch_ns(sample[c]):
                    if unit == "ns":
                        ns_rel.append(c)
                    continue
                ns_dur.append(c)
                continue
            numeric.append(c)
        return CsvSpec(sep, time_col, unit, numeric, ns_rel, ns_dur, split_col), sample

    def clock_for(self, rel: str, spec: CsvSpec) -> str:
        if spec.unit in ("s", "row"):
            return CLOCK_RELATIVE
        return guess_clock(rel)

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        try:
            spec, sample = self.analyse(path)
        except EmptyFile:
            return [Stream(root, rel, path.name, self.kind, guess_clock(rel), self, empty=True)]
        clock = self.clock_for(rel, spec)
        t_first = None
        if spec.time_col is not None and len(sample):
            t = time_to_ns(sample[spec.time_col].head(20), spec.unit if spec.unit != "row" else "s")
            t = t[t != NS_INVALID]
            if len(t):
                t_first = int(t.min())
        base = dict(root=root, rel=rel, kind=self.kind, clock=clock, loader=self,
                    columns=spec.out_columns(), t_first_ns=t_first)
        streams = []
        if spec.split_col:
            vals = sample[spec.split_col].astype(str).str.strip()
            for v in sorted(vals[vals != ""].unique(), key=lambda x: (len(x), x)):
                s = Stream(name=f"{path.name} [{spec.split_col}={v}]", split_value=v, sub=f"{spec.split_col}={v}", **base)
                streams.append(s)
        else:
            streams.append(Stream(name=path.name, **base))
        for s in streams:
            s.info["spec"] = spec
            s.columns = spec.out_columns() + self.extra_names(s)
            s.info["rows_estimate"] = estimate_rows(path, sample)
            s.empty = len(sample) == 0
            self.decorate(s, spec, sample)
        return streams

    def decorate(self, stream: Stream, spec: CsvSpec, sample: pd.DataFrame) -> None:
        pass

    def extra_columns(self, stream: Stream, spec: CsvSpec, chunk: pd.DataFrame, t: np.ndarray) -> dict:
        return {}

    def transform_time(self, stream: Stream, t: np.ndarray, store) -> np.ndarray:
        return t

    def write(self, stream: Stream, out: Path, progress=None, store=None) -> None:
        try:
            self._write_arrow(stream, out, progress, store)
            return
        except Exception:
            if out.exists():
                out.unlink()
        self._write_pandas(stream, out, progress, store)

    def _schema(self, stream: Stream) -> pa.Schema:
        spec: CsvSpec = stream.info["spec"]
        fields = [pa.field(TIME_COL, pa.int64())]
        fields += [pa.field(c, pa.float64()) for c in spec.out_columns() + list(self.extra_names(stream))]
        if spec.split_col:
            fields.append(pa.field(SPLIT_COL, pa.string()))
        return pa.schema(fields)

    def _write_arrow(self, stream: Stream, out: Path, progress=None, store=None) -> None:
        import pyarrow.csv as pacsv
        spec: CsvSpec = stream.info["spec"]
        if spec.time_col is None:
            raise ValueError("no time column")
        types = {c: pa.float64() for c in spec.numeric}
        for c in spec.ns_rel + spec.ns_dur:
            types[c] = pa.int64()
        types[spec.time_col] = pa.int64() if spec.unit == "ns" else pa.float64()
        if spec.split_col:
            types[spec.split_col] = pa.string()
        include = list(types)
        schema = self._schema(stream)
        extra_names = list(self.extra_names(stream))
        reader = pacsv.open_csv(
            stream.path,
            read_options=pacsv.ReadOptions(block_size=32 << 20),
            parse_options=pacsv.ParseOptions(delimiter=spec.sep, invalid_row_handler=lambda row: "skip"),
            convert_options=pacsv.ConvertOptions(column_types=types, include_columns=include,
                                                 null_values=["", "nan", "NaN", "NAN", "None", "null", "inf", "-inf"],
                                                 strings_can_be_null=True))
        writer = pq.ParquetWriter(out, schema)
        prev_last = None
        done = 0
        try:
            for batch in reader:
                tb = batch.column(spec.time_col)
                if spec.unit == "ns":
                    t = tb.to_numpy(zero_copy_only=False) if tb.null_count == 0 else \
                        tb.fill_null(NS_INVALID).to_numpy()
                    t = t.astype(np.int64, copy=False)
                else:
                    f = tb.to_numpy(zero_copy_only=False).astype(np.float64) * UNIT_NS.get(spec.unit, 1_000_000_000)
                    t = np.full(len(f), NS_INVALID, dtype=np.int64)
                    good = np.isfinite(f)
                    t[good] = np.round(f[good]).astype(np.int64)
                ok = t != NS_INVALID
                if spec.unit in ("ns", "us", "ms", "s_epoch"):
                    ok &= t > 10 ** 17
                idx = None if ok.all() else np.flatnonzero(ok)
                t = t if idx is None else t[idx]
                t = self.transform_time(stream, t, store)

                def col(name, kind=np.float64):
                    a = batch.column(name)
                    if kind is np.int64:
                        v = a.fill_null(0).to_numpy().astype(np.int64)
                    else:
                        v = a.to_numpy(zero_copy_only=False).astype(np.float64)
                    return v if idx is None else v[idx]
                data = {TIME_COL: pa.array(t, type=pa.int64())}
                for c in spec.numeric:
                    data[c] = pa.array(col(c))
                for c in spec.ns_rel:
                    v = col(c, np.int64)
                    r = (v - t).astype(np.float64) / 1e6
                    r[v == 0] = np.nan
                    data[rel_name(c)] = pa.array(r)
                for c in spec.ns_dur:
                    v = col(c, np.int64).astype(np.float64) / 1e6
                    data[dur_name(c)] = pa.array(v)
                extra = self.extra_columns(stream, spec, None, t)
                if prev_last is not None and "frame_dt [ms]" in extra and len(t):
                    extra["frame_dt [ms]"][0] = (t[0] - prev_last) / 1e6
                for c in extra_names:
                    data[c] = pa.array(extra.get(c, np.full(len(t), np.nan)), type=pa.float64())
                if spec.split_col:
                    sc = batch.column(spec.split_col)
                    sv = np.asarray(sc.fill_null("").to_pylist(), dtype=object)
                    sv = sv if idx is None else sv[idx]
                    data[SPLIT_COL] = pa.array([str(x).strip() for x in sv], type=pa.string())
                writer.write_table(pa.table(data, schema=schema))
                if len(t):
                    prev_last = int(t[-1])
                done += batch.num_rows
                if progress:
                    est = stream.info.get("rows_estimate") or 1
                    progress(stream, min(done / est, 0.99))
        finally:
            writer.close()

    def _write_pandas(self, stream: Stream, out: Path, progress=None, store=None) -> None:
        spec: CsvSpec = stream.info["spec"]
        cols = spec.out_columns()
        extra_names = list(self.extra_names(stream))
        fields = [pa.field(TIME_COL, pa.int64())] + [pa.field(c, pa.float64()) for c in cols + extra_names]
        if spec.split_col:
            fields.append(pa.field(SPLIT_COL, pa.string()))
        schema = pa.schema(fields)
        writer = pq.ParquetWriter(out, schema)
        total = stream.info.get("rows_estimate") or 1
        done = 0
        prev_last = None
        try:
            reader = pd.read_csv(stream.path, sep=spec.sep, dtype=str, keep_default_na=False,
                                 on_bad_lines="skip", engine="c", chunksize=CHUNK_ROWS)
            for chunk in reader:
                if spec.time_col is None:
                    t = (np.arange(len(chunk), dtype=np.int64) + done) * 1_000_000_000
                else:
                    t = time_to_ns(chunk[spec.time_col], spec.unit if spec.unit != "row" else "s")
                ok = t != NS_INVALID
                if spec.unit in ("ns", "us", "ms", "s_epoch") and stream.info.get("epoch_time", True):
                    ok &= t > 10 ** 17
                chunk = chunk.loc[ok]
                t = t[ok]
                t = self.transform_time(stream, t, store)
                data = {TIME_COL: pa.array(t, type=pa.int64())}
                for c in spec.numeric:
                    data[c] = pa.array(to_float(chunk[c]) if c in chunk else np.full(len(t), np.nan))
                for c in spec.ns_rel:
                    v = to_ns(chunk[c])
                    r = (v - t).astype(np.float64) / 1e6
                    r[v == NS_INVALID] = np.nan
                    r[v == 0] = np.nan
                    data[rel_name(c)] = pa.array(r)
                for c in spec.ns_dur:
                    data[dur_name(c)] = pa.array(to_float(chunk[c]) / 1e6)
                extra = self.extra_columns(stream, spec, chunk, t)
                if prev_last is not None and "frame_dt [ms]" in extra and len(t):
                    extra["frame_dt [ms]"][0] = (t[0] - prev_last) / 1e6
                for c in extra_names:
                    data[c] = pa.array(extra.get(c, np.full(len(t), np.nan)), type=pa.float64())
                if spec.split_col:
                    data[SPLIT_COL] = pa.array(chunk[spec.split_col].astype(str).str.strip().tolist(),
                                               type=pa.string())
                writer.write_table(pa.table(data, schema=schema))
                if len(t):
                    prev_last = int(t[-1])
                done += len(chunk)
                if progress:
                    progress(stream, min(done / total, 0.99))
        except pd.errors.EmptyDataError:
            pass
        finally:
            writer.close()

    def extra_names(self, stream: Stream) -> list[str]:
        return []


class EmptyFile(Exception):
    pass


def estimate_rows(path: Path, sample: pd.DataFrame) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if len(sample) == 0:
        return 0
    _, lines = read_head(path, 64 * 1024)
    if len(lines) < 2:
        return len(sample)
    avg = sum(len(x) + 1 for x in lines[1:]) / max(1, len(lines) - 1)
    return int(size / max(avg, 1))
