"""Loaders for the operator-PC orchestrator logs: orchestrator.hdf5 (RunLogger) and nudge.csv (teleop-nudge EventLog)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..model import CLOCK_AVATAR, EVENTS, SIGNALS, Stream
from .base import Loader, table_from
from ..intent_names import column_labels
from .csv_table import CsvLoader


def _flat_columns(name: str, shape: tuple) -> list[str]:
    """Column names for a dataset of shape (n, *dims): name, name_i or name_i_j."""
    if len(shape) == 1:
        return [name]
    idx = np.ndindex(*shape[1:])
    return [name + "".join(f"_{k}" for k in ix) for ix in idx]


class LocalHdf5Loader(Loader):
    """frames/* and modules/** of an orchestrator.hdf5 / run.hdf5 as one signal stream per group, nested under the file in the tree."""
    version = "1"
    names = {"orchestrator.hdf5", "local.hdf5", "run.hdf5"}

    def match(self, path: Path) -> bool:
        return path.name.lower() in self.names or path.name.lower().startswith(("local_", "run_")) and path.suffix == ".hdf5"

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        import h5py
        streams = []
        with h5py.File(path, "r") as f:
            if "frames" not in f or "timestamp_ns" not in f["frames"]:
                return [Stream(root, rel, path.name, SIGNALS, CLOCK_AVATAR, self, empty=True)]
            t = f["frames/timestamp_ns"]
            t_first = int(t[0]) if len(t) else None
            groups: dict[str, list[str]] = {}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name != "frames/timestamp_ns":
                    groups.setdefault(str(Path(name).parent.as_posix()), []).extend(_flat_columns(Path(name).name, obj.shape))
            f.visititems(visit)
            labels = {}
            for group, cols in groups.items():
                if group.startswith("modules/"):
                    try:
                        labels[group] = column_labels(f, path.parent, group, cols)
                    except Exception as e:
                        labels[group] = ({}, {}, f"failed: {e}")
        for group, cols in sorted(groups.items()):
            short = group[len("modules/"):] if group.startswith("modules/") else group
            s = Stream(root, rel, short, SIGNALS, CLOCK_AVATAR, self, columns=cols, sub=group, t_first_ns=t_first)
            s.info["group"] = group
            s.info["cache_id"] = group
            s.info["file_group"] = True
            if group in labels and labels[group][0]:
                s.info["col_labels"], s.info["col_colors"], s.info["names_source"] = labels[group]
                s.info["display_name"] = group.replace("modules/", "").replace("/", " ")
                s.info["note"] = f"candidate names from {labels[group][2]}"
            s.empty = t_first is None
            streams.append(s)
        return streams

    def build(self, stream: Stream, store=None) -> pa.Table:
        import h5py
        cols = {}
        with h5py.File(stream.path, "r") as f:
            t = np.asarray(f["frames/timestamp_ns"][:], dtype=np.int64)
            n = len(t)
            for name, ds in f[stream.info["group"]].items():
                if not isinstance(ds, h5py.Dataset) or ds.shape[0] != n or f"{stream.info['group']}/{name}" == "frames/timestamp_ns":
                    continue
                a = np.asarray(ds[:], dtype=np.float64).reshape(n, -1)
                for k, c in enumerate(_flat_columns(name, ds.shape)):
                    cols[c] = a[:, k]
        return table_from(t, cols)


class NudgeLoader(CsvLoader):
    """nudge.csv twice: numeric columns as signals, and every issued cue as an event marker."""
    version = "1"
    prefer_time = ["t_ns"]

    def match(self, path: Path) -> bool:
        return path.name.lower() == "nudge.csv" or path.name.lower().startswith("nudge_") and path.suffix == ".csv"

    def clock_for(self, rel: str, spec) -> str:
        return CLOCK_AVATAR

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        streams = super().discover(root, path, rel)
        ev = Stream(root, rel, f"{path.name} [issued]", EVENTS, CLOCK_AVATAR, self, sub="issued",
                    t_first_ns=streams[0].t_first_ns, empty=streams[0].empty)
        ev.info["cache_id"] = "issued"
        return streams + [ev]

    def write(self, stream: Stream, out: Path, progress=None, store=None) -> None:
        if stream.sub == "issued":
            pq.write_table(self.build(stream, store), out)
        else:
            super().write(stream, out, progress, store)

    def build(self, stream: Stream, store=None) -> pa.Table:
        if stream.sub != "issued":
            raise NotImplementedError("signal streams go through CsvLoader.write")
        times, types, labels = [], [], []
        with open(stream.path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("outcome") != "issued":
                    continue
                times.append(int(row["t_ns"]))
                types.append(row["nudge"])
                labels.append(f"{row['nudge']} {row['arm']} {row['channel']}/{row['primitive']} i={row['intensity']}")
        return table_from(np.asarray(times, dtype=np.int64), {}, strings={"type": types, "label": labels})
