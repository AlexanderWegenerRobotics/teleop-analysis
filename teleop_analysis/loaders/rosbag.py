from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..model import SIGNALS, TIME_COL, Stream
from .base import Loader, guess_clock

SKIP_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/PointCloud2",
              "sensor_msgs/msg/CameraInfo", "tf2_msgs/msg/TFMessage")
MAX_ARRAY = 32
BATCH = 50_000


def _typestore():
    from rosbags.typesys import Stores, get_typestore
    return get_typestore(Stores.ROS2_HUMBLE)


def _flatten(obj, prefix: str, out: dict):
    if isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out[prefix] = float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biuf" and obj.size <= MAX_ARRAY:
            for i, v in enumerate(obj.ravel()):
                out[f"{prefix}[{i}]"] = float(v)
    elif isinstance(obj, (list, tuple)):
        if len(obj) <= MAX_ARRAY:
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}[{i}]", out)
    elif hasattr(obj, "__dataclass_fields__"):
        for k in obj.__dataclass_fields__:
            if k == "__msgtype__":
                continue
            _flatten(getattr(obj, k), f"{prefix}.{k}" if prefix else k, out)


def _stamp_ns(msg) -> int | None:
    h = getattr(msg, "header", None)
    st = getattr(h, "stamp", None)
    if st is None:
        return None
    ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
    return ns if ns > 0 else None


class RosbagLoader(Loader):
    version = "1"

    def match(self, path: Path) -> bool:
        return path.suffix.lower() == ".db3"

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        try:
            ts = _typestore()
        except ImportError:
            root.errors.append(f"{rel}: install 'rosbags' to read ROS2 bags")
            return []
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            topics = con.execute("SELECT id, name, type FROM topics ORDER BY name").fetchall()
            streams = []
            for tid, name, mtype in topics:
                s = Stream(root, rel, f"{path.name} {name}", SIGNALS, guess_clock(rel), self, sub=name)
                s.info.update(topic_id=tid, topic=name, msgtype=mtype, cache_id=f"topic{tid}")
                if mtype in SKIP_TYPES:
                    s.info["note"] = f"{mtype} skipped (not a numeric signal)"
                    s.empty = True
                    streams.append(s)
                    continue
                row = con.execute("SELECT timestamp, data FROM messages WHERE topic_id=? LIMIT 1", (tid,)).fetchone()
                if row is None:
                    s.empty = True
                    streams.append(s)
                    continue
                try:
                    msg = ts.deserialize_cdr(row[1], mtype)
                except Exception as e:
                    s.error = f"cannot decode {mtype}: {e}"
                    streams.append(s)
                    continue
                flat = {}
                _flatten(msg, "", flat)
                s.columns = [k for k in flat if not k.startswith("header.")]
                s.info["fields"] = s.columns
                st = _stamp_ns(msg)
                s.info["use_header"] = st is not None
                s.t_first_ns = st if st is not None else int(row[0])
                streams.append(s)
            return streams
        finally:
            con.close()

    def write(self, stream: Stream, out: Path, progress=None, store=None) -> None:
        ts = _typestore()
        fields = stream.info.get("fields", [])
        schema = pa.schema([pa.field(TIME_COL, pa.int64())] + [pa.field(c, pa.float64()) for c in fields])
        writer = pq.ParquetWriter(out, schema)
        con = sqlite3.connect(f"file:{stream.path}?mode=ro", uri=True)
        try:
            total = con.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?",
                                (stream.info["topic_id"],)).fetchone()[0] or 1
            cur = con.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                              (stream.info["topic_id"],))
            done = 0
            while True:
                rows = cur.fetchmany(BATCH)
                if not rows:
                    break
                t = np.empty(len(rows), dtype=np.int64)
                cols = {c: np.full(len(rows), np.nan) for c in fields}
                for i, (stamp, data) in enumerate(rows):
                    try:
                        msg = ts.deserialize_cdr(data, stream.info["msgtype"])
                    except Exception:
                        t[i] = stamp
                        continue
                    h = _stamp_ns(msg) if stream.info.get("use_header") else None
                    t[i] = h if h is not None else stamp
                    flat = {}
                    _flatten(msg, "", flat)
                    for k, v in flat.items():
                        if k in cols:
                            cols[k][i] = v
                data = {TIME_COL: pa.array(t)}
                data.update({k: pa.array(v) for k, v in cols.items()})
                writer.write_table(pa.table(data, schema=schema))
                done += len(rows)
                if progress:
                    progress(stream, min(done / total, 0.99))
        finally:
            con.close()
            writer.close()
