from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from . import math_ops
from .cache import TableStore
from .loaders import discover_root
from .model import (CLOCK_AVATAR, CLOCK_INTERFACE, CLOCK_RELATIVE, EVENTS, SIGNALS, VIDEO, Root, Stream,
                    split_signal_key)

DERIVED = "derived"

ROBOT_HOLD_MS = 9.0


class Catalog:
    def __init__(self):
        self.roots: list[Root] = []
        self.store = TableStore()
        self.user_offsets_ms: dict[tuple[str, str], float] = {}
        self.estimated_offsets_ms: dict[tuple[str, str], float | None] = {}
        self.shifts_s: dict[tuple[str, str], float] = {}
        self.origin_ns: int | None = None
        self._streams: dict[str, Stream] = {}
        self._order: dict[str, np.ndarray | None] = {}
        self._anchor: dict[tuple[str, str], int | None] = {}
        self.derived_root = Root(path=Path("."), name=DERIVED)
        self.derived: dict[str, dict] = {}
        self._derived_cache: dict[str, tuple] = {}

    def add_root(self, path, name: str | None = None) -> Root:
        path = Path(path)
        for r in self.roots:
            if r.path.resolve() == path.resolve():
                return r
        base = name or path.name or str(path)
        names = {r.name for r in self.roots}
        nm, i = base, 2
        while nm in names:
            nm, i = f"{base} ({i})", i + 1
        root = discover_root(path, nm)
        self.roots.append(root)
        self._index()
        if self.origin_ns is None:
            firsts = [s.t_first_ns for s in root.streams if s.t_first_ns and s.clock != CLOCK_RELATIVE]
            self.origin_ns = int(min(firsts)) if firsts else 0
        return root

    def remove_root(self, name: str) -> None:
        self.roots = [r for r in self.roots if r.name != name]
        self._index()

    def reload(self) -> None:
        old = [(r.path, r.name) for r in self.roots]
        self.roots = []
        self.store.clear()
        self._order.clear()
        self._anchor.clear()
        self.estimated_offsets_ms.clear()
        self._derived_cache.clear()
        for p, n in old:
            if Path(p).exists():
                root = discover_root(Path(p), n)
                self.roots.append(root)
        self._index()

    def _index(self) -> None:
        self._streams = {s.key: s for r in self.roots + [self.derived_root] for s in r.streams}

    def _unique_name(self, base: str) -> str:
        base = base.replace("::", ":").replace("#", "").replace("/", " ")
        name, i = base, 2
        while name in self.derived:
            name, i = f"{base} ({i})", i + 1
        return name

    def add_derived(self, op: str, inputs: list[str], params: dict | None = None, name: str | None = None) -> list[str]:
        params = dict(params or {})
        if not name:
            name = self._auto_name(op, inputs, params)
        name = self._unique_name(name)
        d = {"name": name, "op": op, "inputs": list(inputs), "params": params}
        self.derived[name] = d
        try:
            t, cols = self._eval_derived(name)
        except Exception:
            self.derived.pop(name, None)
            self._derived_cache.pop(name, None)
            raise
        s = Stream(self.derived_root, name, name, SIGNALS, DERIVED, None, columns=list(cols))
        s.info["derived"] = d
        s.info["note"] = f"{math_ops.OPS[op].label} of " + ", ".join(inputs)
        self.derived_root.streams.append(s)
        self._index()
        return [s.signal_key(c) for c in cols]

    def _auto_name(self, op: str, inputs: list[str], params: dict) -> str:
        import re
        cols = [split_signal_key(k)[1] or k for k in inputs]
        streams = [self.stream(k) for k in inputs]
        base = ""
        s0 = streams[0]
        if s0 is not None and s0.root is not self.derived_root:
            stem = re.sub(r"\.(csv|tsv|jsonl|db3)$", "", s0.name)
            folder = s0.folder.split("/")[-1] if s0.folder else ""
            same = all(s is s0 for s in streams)
            base = f"{folder} {stem}".strip() if same else folder
        elif s0 is not None:
            base = s0.rel
        m = [re.match(r"^(.*?)(\d+)$", c) for c in cols]
        if len(cols) > 1 and all(m) and len({x.group(1) for x in m}) == 1:
            nums = [x.group(2) for x in m]
            short = f"{m[0].group(1).rstrip('_')}" if op == "se3" else f"{m[0].group(1)}{nums[0]}..{nums[-1]}"
        elif len(cols) <= 3:
            short = ", ".join(cols)
        else:
            short = f"{cols[0]} … {cols[-1]}"
        if s0 is not None and s0.root is self.derived_root:
            short = ""
        what = f"{base} {short}".strip()
        label = {"se3": "pose", "quat2rpy": "rpy", "norm": "‖·‖", "sum": "sum", "diff": "a−b", "mul": "a·b",
                 "div": "a/b", "derivative": "d/dt", "integral": "∫dt", "movavg": "avg", "abs": "abs",
                 "scale": "scaled", "shift": "shifted", "unwrap": "unwrap", "rad2deg": "deg", "deg2rad": "rad"}
        if op == "expr":
            name = f"{params.get('expression', 'expr')} [{what}]"
        elif op == "se3":
            name = f"{what} pose"
        elif op == "shift":
            name = f"{what} shifted {float(params.get('seconds', 0)) * 1000:+.1f} ms"
        else:
            name = f"{label.get(op, op)} {what}"
        return name.replace("/", " ").replace("::", ":").strip()

    def remove_derived(self, name: str) -> None:
        self.derived.pop(name, None)
        self._derived_cache.pop(name, None)
        self.derived_root.streams = [s for s in self.derived_root.streams if s.rel != name]
        self._index()

    def derived_defs(self) -> list[dict]:
        return [dict(d) for d in self.derived.values()]

    def _eval_derived(self, name: str):
        if name in self._derived_cache:
            return self._derived_cache[name]
        d = self.derived[name]
        ts, ys, names = [], [], []
        for k in d["inputs"]:
            t, y = self.get(k)
            ts.append(t)
            ys.append(y)
            names.append(split_signal_key(k)[1] or k)
        t, cols = math_ops.evaluate(d["op"], ts, ys, names, d["params"])
        if len(t) > 1 and np.any(np.diff(t) < 0):
            o = np.argsort(t, kind="stable")
            t = t[o]
            cols = {k: v[o] for k, v in cols.items()}
        self._derived_cache[name] = (t, cols)
        return t, cols

    def root(self, name: str) -> Root | None:
        return next((r for r in self.roots if r.name == name), None)

    def streams(self) -> list[Stream]:
        return list(self._streams.values())

    def stream(self, key: str) -> Stream | None:
        return self._streams.get(split_signal_key(key)[0])

    def estimate_offset_ms(self, root: Root, clock: str) -> float | None:
        k = (root.name, clock)
        if k in self.estimated_offsets_ms:
            return self.estimated_offsets_ms[k]
        val = None
        if clock == CLOCK_AVATAR:
            ests = []
            for s in root.streams:
                if PurePosixPath(s.rel).name != "command.csv" or s.empty or s.error:
                    continue
                if s.split_value not in (None, "0"):
                    continue
                need = ["rtt_ms", "robot_recv - t [ms]", "robot_tx - t [ms]"]
                if not all(c in s.columns for c in need):
                    continue
                try:
                    rtt = self.store.column(s, "rtt_ms")
                    rx = self.store.column(s, "robot_recv - t [ms]")
                    tx = self.store.column(s, "robot_tx - t [ms]")
                except Exception:
                    continue
                ok = np.isfinite(rtt) & np.isfinite(rx) & np.isfinite(tx) & (rtt > 0)
                if ok.sum() < 100:
                    continue
                one_way = (np.median(rtt[ok]) - ROBOT_HOLD_MS) / 2.0
                nuc_minus_pc = one_way - np.median(rx[ok] - tx[ok])
                ests.append(-float(nuc_minus_pc))
            if ests:
                val = float(np.median(ests))
        self.estimated_offsets_ms[k] = val
        return val

    def offset_ms(self, root: Root, clock: str) -> float:
        k = (root.name, clock)
        if k in self.user_offsets_ms:
            return self.user_offsets_ms[k]
        if clock in (CLOCK_INTERFACE, CLOCK_RELATIVE):
            return 0.0
        est = self.estimate_offset_ms(root, clock)
        return est if est is not None else 0.0

    def shift_s(self, stream: Stream) -> float:
        folder = stream.folder
        best, best_len = 0.0, -1
        for (rn, f), v in self.shifts_s.items():
            if rn != stream.root.name:
                continue
            if f == "" or folder == f or folder.startswith(f + "/"):
                if len(f) > best_len:
                    best, best_len = v, len(f)
        return best

    def _relative_anchor(self, stream: Stream) -> int:
        folder = stream.folder
        k = (stream.root.name, folder)
        if k not in self._anchor:
            val = None
            parts = folder.split("/") if folder else []
            while val is None:
                f = "/".join(parts)
                cands = [s for s in stream.root.streams
                         if s.folder == f and s.clock != CLOCK_RELATIVE and s.t_first_ns]
                if cands:
                    val = min(int(s.t_first_ns + self.offset_ms(s.root, s.clock) * 1e6) for s in cands)
                if not parts:
                    break
                parts = parts[:-1]
            self._anchor[k] = val
        v = self._anchor[k]
        return v if v is not None else (self.origin_ns or 0)

    def to_seconds(self, stream: Stream, t_ns: np.ndarray) -> np.ndarray:
        origin = self.origin_ns or 0
        if stream.clock == CLOCK_RELATIVE:
            base = self._relative_anchor(stream)
        else:
            base = int(round(self.offset_ms(stream.root, stream.clock) * 1e6))
        return (t_ns - (origin - base)).astype(np.float64) / 1e9 + self.shift_s(stream)

    def stream_start_s(self, stream: Stream) -> float | None:
        if stream.t_first_ns is None or stream.clock == CLOCK_RELATIVE:
            try:
                t = self.store.time_ns(stream)
            except Exception:
                return None
            if not len(t):
                return None
            return float(self.to_seconds(stream, t[:1])[0])
        return float(self.to_seconds(stream, np.array([stream.t_first_ns], dtype=np.int64))[0])

    def _sorted(self, stream: Stream):
        k = stream.key
        if k not in self._order:
            t = self.store.time_ns(stream)
            self._order[k] = None if len(t) < 2 or np.all(np.diff(t) >= 0) else np.argsort(t, kind="stable")
        return self._order[k]

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        skey, col = split_signal_key(key)
        s = self._streams.get(skey)
        if s is None:
            raise KeyError(f"no stream {skey}")
        if s.root is self.derived_root:
            t, cols = self._eval_derived(s.rel)
            if col not in cols:
                raise KeyError(f"{col} not in {s.rel}")
            return t, cols[col]
        t = self.store.time_ns(s)
        y = self.store.column(s, col)
        o = self._sorted(s)
        if o is not None:
            t, y = t[o], y[o]
        return self.to_seconds(s, t), y

    def events(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self._streams[split_signal_key(key)[0]]
        t = self.store.time_ns(s)
        if not len(t):
            return np.zeros(0), np.zeros(0, dtype=object), np.zeros(0, dtype=object)
        return self.to_seconds(s, t), self.store.column(s, "type"), self.store.column(s, "label")

    def video_frames(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        s = self._streams[split_signal_key(key)[0]]
        t = self.store.time_ns(s)
        names = self.store.available(s)
        if "__row" in names:
            idx = self.store.column(s, "__row")
        elif "frame_idx" in names:
            idx = self.store.column(s, "frame_idx")
        else:
            idx = np.arange(len(t), dtype=np.float64)
        o = self._sorted(s)
        if o is not None:
            t, idx = t[o], idx[o]
        return self.to_seconds(s, t), idx

    def invalidate_time(self) -> None:
        self._anchor.clear()
        self._derived_cache.clear()

    def find(self, query: str, kinds=None, limit: int = 2000) -> list[str]:
        toks = [t.lower() for t in query.split() if t]
        out = []
        for s in self._streams.values():
            if kinds and s.kind not in kinds:
                continue
            base = f"{s.key} {s.name}".lower()
            if s.kind in (EVENTS, VIDEO) and all(t in base for t in toks):
                out.append(s.key)
            nice = s.info.get("col_labels") or {}
            disp = s.info.get("display_name", "")
            for c in s.columns:
                if all(t in f"{base} {disp} {c} {nice.get(c, '')}".lower() for t in toks):
                    out.append(s.signal_key(c))
                    if len(out) >= limit:
                        return out
        return out

    def series(self, key: str) -> pd.Series:
        t, y = self.get(key)
        return pd.Series(y, index=pd.Index(t, name="t_s"), name=split_signal_key(key)[1])

    def frame(self, stream_key: str, columns: list[str] | None = None) -> pd.DataFrame:
        s = self._streams[stream_key]
        if s.root is self.derived_root:
            t, cols = self._eval_derived(s.rel)
            return pd.DataFrame({c: cols[c] for c in (columns or s.columns)}, index=pd.Index(t, name="t_s"))
        cols = columns or s.columns
        t = self.store.time_ns(s)
        o = self._sorted(s)
        data = {c: self.store.column(s, c) for c in cols}
        if o is not None:
            t = t[o]
            data = {k: v[o] for k, v in data.items()}
        df = pd.DataFrame(data, index=pd.Index(self.to_seconds(s, t), name="t_s"))
        df.insert(0, "t_ref_ns", t + (0 if s.clock == CLOCK_RELATIVE else int(self.offset_ms(s.root, s.clock) * 1e6)))
        return df


def open_folder(*paths) -> Catalog:
    cat = Catalog()
    for p in paths:
        cat.add_root(p)
    return cat
