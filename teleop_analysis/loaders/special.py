from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from ..model import CLOCK_INTERFACE, CLOCK_OPERATOR, CLOCK_RELATIVE, EVENTS, SIGNALS, VIDEO, Stream
from .base import NS_INVALID, Loader, guess_clock, read_head, sniff_sep, table_from
from .csv_table import CsvLoader, CsvSpec


def find_video(folder: Path, stem: str) -> dict:
    names = [stem]
    if stem.startswith("video_"):
        names.append(stem[len("video_"):])
    for ext in (".mp4", ".mkv", ".avi", ".mov"):
        for n in names:
            p = folder / f"{n}{ext}"
            if p.exists():
                return {"path": p, "type": "cv"}
    for n in names:
        p = folder / f"{n}.h264"
        if p.exists():
            return {"path": p, "type": "h264"}
    for n in names:
        for cand in (folder / f"{n}.hdf5", folder / f"{n}.h5"):
            if cand.exists():
                return {"path": cand, "type": "h5"}
    return {}


class VideoTimestampsLoader(CsvLoader):
    version = "4"
    kind = VIDEO
    prefer_time = ["capture_time_ns", "wall_clock_ns"]

    def match(self, path: Path) -> bool:
        return path.name.lower().endswith(".timestamps.csv")

    def clock_for(self, rel: str, spec: CsvSpec) -> str:
        if spec.unit in ("s", "row"):
            return CLOCK_RELATIVE
        if "operator" in rel.lower():
            return CLOCK_OPERATOR
        return guess_clock(rel)

    def decorate(self, stream: Stream, spec: CsvSpec, sample: pd.DataFrame) -> None:
        if "timestamp_domain" in sample.columns:
            stream.clock = CLOCK_OPERATOR
        stem = stream.path.name[: -len(".timestamps.csv")]
        stream.video = find_video(stream.path.parent, stem)
        stream.name = stem
        if not stream.video:
            stream.kind = SIGNALS
            stream.info["note"] = "no matching video file found"

    def extra_names(self, stream: Stream) -> list[str]:
        return ["frame_dt [ms]"]

    def extra_columns(self, stream, spec, chunk, t):
        dt = np.full(len(t), np.nan)
        if len(t) > 1:
            dt[1:] = np.diff(t) / 1e6
        return {"frame_dt [ms]": dt}


class WrenchTruthLoader(CsvLoader):
    version = "4"
    prefer_time = ["sim_time"]

    def match(self, path: Path) -> bool:
        return path.name.lower().endswith("_wrench_truth.csv")

    def decorate(self, stream: Stream, spec: CsvSpec, sample: pd.DataFrame) -> None:
        sib = stream.path.with_name(stream.path.name.replace("_wrench_truth.csv", ".csv"))
        if not sib.exists():
            stream.info["note"] = "no sibling arm log, kept on simulation time"
            return
        _, lines = read_head(sib, 64 * 1024)
        if not lines:
            return
        header = lines[0].split(sniff_sep(lines[0]))
        if "sim_time" in header and "wall_clock_ns" in header:
            stream.deps = [sib]
            stream.info["sibling_rel"] = str(Path(stream.rel).with_name(sib.name).as_posix())
            stream.clock = guess_clock(stream.rel)
            stream.t_first_ns = None
            stream.info["note"] = f"sim_time mapped to wall clock via {sib.name}"

    def transform_time(self, stream: Stream, t: np.ndarray, store) -> np.ndarray:
        rel = stream.info.get("sibling_rel")
        if not rel or store is None or not len(t):
            return t
        sib = next((s for s in stream.root.streams if s.rel == rel and s.split_value is None), None)
        if sib is None:
            return t
        sim = store.column(sib, "sim_time")
        wall = store.time_ns(sib)
        ok = np.isfinite(sim)
        sim, wall = sim[ok], wall[ok]
        if len(sim) < 2:
            return t
        order = np.argsort(sim, kind="stable")
        sim, wall = sim[order], wall[order]
        sim, idx = np.unique(sim, return_index=True)
        wall = wall[idx]
        base = wall[0]
        w = (wall - base).astype(np.float64)
        x = t.astype(np.float64) / 1e9
        y = np.interp(x, sim, w)
        if len(sim) >= 2:
            lo = x < sim[0]
            hi = x > sim[-1]
            k0 = (w[min(50, len(w) - 1)] - w[0]) / max(sim[min(50, len(sim) - 1)] - sim[0], 1e-9)
            k1 = (w[-1] - w[max(0, len(w) - 51)]) / max(sim[-1] - sim[max(0, len(sim) - 51)], 1e-9)
            y[lo] = w[0] + (x[lo] - sim[0]) * k0
            y[hi] = w[-1] + (x[hi] - sim[-1]) * k1
        return (base + np.round(y)).astype(np.int64)


class MetaLoader(Loader):
    version = "3"

    def match(self, path: Path) -> bool:
        return path.name.lower().endswith("_meta.csv")

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        s = Stream(root, rel, path.name, EVENTS, CLOCK_RELATIVE, self)
        sib = path.with_name(path.name.replace("_meta.csv", ".csv"))
        if sib.exists():
            s.deps = [sib]
            s.info["sibling_rel"] = str(Path(rel).with_name(sib.name).as_posix())
        _, lines = read_head(path, 8 * 1024)
        s.empty = len(lines) < 2
        return [s]

    def _sibling(self, stream: Stream):
        rel = stream.info.get("sibling_rel")
        return next((x for x in stream.root.streams if x.rel == rel), None)

    def resolve_clock(self, stream: Stream) -> None:
        sib = self._sibling(stream)
        if sib is not None and sib.t_first_ns is not None:
            stream.clock = sib.clock

    def build(self, stream: Stream, store=None) -> pa.Table:
        try:
            df = pd.read_csv(stream.path, sep=sniff_sep(read_head(stream.path, 4096)[1][0]), dtype=str,
                             keep_default_na=False, on_bad_lines="skip")
        except (pd.errors.EmptyDataError, IndexError):
            df = pd.DataFrame()
        sib = self._sibling(stream)
        base = sib.t_first_ns if sib is not None and sib.t_first_ns is not None else 0
        times, types, labels = [], [], []
        for _, r in df.iterrows():
            try:
                ts = float(r.get("time_s", ""))
            except ValueError:
                continue
            ev = r.get("event", "") or "event"
            parts = [f"ep{r.get('episode_id', '')}", ev]
            for k in ("mode", "seed", "ann_label", "ann_atype", "color_bin_mapping"):
                v = str(r.get(k, "")).strip()
                if not v:
                    continue
                if k == "color_bin_mapping" and not v.startswith("{"):
                    parts.append(v)
                else:
                    parts.append(f"{k}={v}")
            times.append(base + int(round(ts * 1e9)))
            types.append(ev)
            labels.append(" ".join(parts))
        return table_from(np.asarray(times, dtype=np.int64), {}, strings={"type": types, "label": labels})


_EV_RE = re.compile(r"^\s*(\d{12,})\s+(\S+)\s*(.*?)\s*$")


class EventsLogLoader(Loader):
    version = "1"

    def match(self, path: Path) -> bool:
        return path.name.lower() == "events.log" or path.name.lower().endswith(".events.log")

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        _, lines = read_head(path, 4096)
        s = Stream(root, rel, path.name, EVENTS, guess_clock(rel), self)
        s.empty = not any(_EV_RE.match(x) for x in lines)
        if "interface" not in rel.lower() and "operator" not in rel.lower():
            s.clock = CLOCK_INTERFACE
        return [s]

    def build(self, stream: Stream, store=None) -> pa.Table:
        times, types, labels = [], [], []
        with open(stream.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _EV_RE.match(line)
                if not m:
                    continue
                times.append(int(m.group(1)))
                types.append(m.group(2))
                labels.append(f"{m.group(2)} {m.group(3)}".strip())
        return table_from(np.asarray(times, dtype=np.int64), {}, strings={"type": types, "label": labels})


def iter_json_objects(text: str):
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            nl = text.find("\n", i)
            if nl < 0:
                break
            i = nl + 1
            continue
        if isinstance(obj, dict):
            yield obj


class FramesJsonlLoader(Loader):
    version = "3"
    time_keys = ("ts", "timestamp", "time")

    def match(self, path: Path) -> bool:
        return path.suffix.lower() in (".jsonl", ".ndjson")

    def discover(self, root, path: Path, rel: str) -> list[Stream]:
        text, _ = read_head(path, 64 * 1024)
        objs = []
        for o in iter_json_objects(text):
            objs.append(o)
            if len(objs) >= 30:
                break
        if not objs:
            return [Stream(root, rel, path.name, SIGNALS, guess_clock(rel), self, empty=True)]
        keys = []
        for o in objs:
            for k, v in o.items():
                if k not in keys and isinstance(v, (int, float, bool)):
                    keys.append(k)
        tkey = next((k for k in self.time_keys if k in keys), None)
        cols = []
        for k in keys:
            if k == tkey:
                continue
            if k.endswith("_ns"):
                cols.append(f"{k[:-3]} - t [ms]")
            else:
                cols.append(k)
        cols.append("frame_dt [ms]")
        t_first = None
        if tkey is not None:
            t_first = int(float(objs[0][tkey]) * (1 if float(objs[0][tkey]) > 1e17 else 1e9))
        clock = guess_clock(rel) if tkey else CLOCK_RELATIVE
        meta = {}
        mp = path.with_name("metadata.json")
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8", errors="replace"))
            except ValueError:
                meta = {}
        streams = []
        videos = []
        for key, default in (("raw_video_file", "session.mp4"), ("attention_video_file", "session_attention.mp4")):
            fname = meta.get(key) or default
            vp = path.with_name(fname)
            if vp.exists() and vp not in [v["path"] for v in videos]:
                videos.append({"path": vp, "type": "cv"})
        for vp in sorted(path.parent.glob("*.mp4")):
            if vp.stem.startswith("session") and vp not in [v["path"] for v in videos]:
                videos.append({"path": vp, "type": "cv"})
        if not videos:
            videos = [{}]
        for i, v in enumerate(videos):
            name = path.name if not v else f"{v['path'].name}"
            s = Stream(root, rel, name, VIDEO if v else SIGNALS, clock, self, columns=list(cols) if i == 0 else [],
                       t_first_ns=t_first, video=v)
            s.info.update(time_key=tkey, keys=keys, meta=meta)
            if i > 0:
                s.sub = v["path"].name
                s.info["video_only"] = True
            streams.append(s)
        return streams

    def build(self, stream: Stream, store=None) -> pa.Table:
        tkey = stream.info["time_key"]
        keys = stream.info["keys"]
        text = stream.path.read_text(encoding="utf-8", errors="replace")
        rows = {k: [] for k in keys}
        tl = []
        for n, o in enumerate(iter_json_objects(text)):
            if tkey is None:
                tl.append(n * 1_000_000_000)
            else:
                try:
                    v = float(o.get(tkey))
                except (TypeError, ValueError):
                    v = float("nan")
                tl.append(int(v if v > 1e17 else v * 1e9) if np.isfinite(v) else NS_INVALID)
            for k in keys:
                v = o.get(k)
                rows[k].append(float(v) if isinstance(v, (int, float, bool)) else np.nan)
        t = np.asarray(tl, dtype=np.int64)
        cols = {}
        for k in keys:
            if k == tkey:
                continue
            a = np.asarray(rows[k], dtype=np.float64)
            if k.endswith("_ns"):
                r = (a - t.astype(np.float64)) / 1e6
                r[~np.isfinite(a) | (a == 0)] = np.nan
                cols[f"{k[:-3]} - t [ms]"] = r
            else:
                cols[k] = a
        dt = np.full(len(t), np.nan)
        if len(t) > 1:
            dt[1:] = np.diff(t) / 1e6
        cols["frame_dt [ms]"] = dt
        keep = t != NS_INVALID
        cols = {k: v[keep] for k, v in cols.items()}
        cols["__row"] = np.flatnonzero(keep).astype(np.float64)
        return table_from(t[keep], cols)
