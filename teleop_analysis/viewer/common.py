from __future__ import annotations

import json
import re

import numpy as np

MIME = "application/x-teleop-keys"
MOVE_MIME = "application/x-teleop-move"
STYLES = {"solid": "solid", "dash": "dashed", "dot": "dotted", "dashdot": "dash-dot"}

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf"]

EVENT_COLORS = ["#7f7f7f", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#1f77b4", "#8c564b", "#e377c2"]


def keys_mime(keys: list[str], move: bool = False):
    from PySide6.QtCore import QMimeData
    m = QMimeData()
    m.setData(MIME, json.dumps(keys).encode())
    if move:
        m.setData(MOVE_MIME, b"1")
    m.setText("\n".join(keys))
    return m


def mime_keys(mime) -> list[str]:
    if mime.hasFormat(MIME):
        try:
            return list(json.loads(bytes(mime.data(MIME)).decode()))
        except ValueError:
            return []
    return []


def event_color(kind: str) -> str:
    return EVENT_COLORS[sum(map(ord, kind or "")) % len(EVENT_COLORS)]


def with_gaps(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(t) < 3:
        return t, y
    dt = np.diff(t)
    pos = dt[dt > 0]
    if not len(pos):
        return t, y
    thr = max(10 * float(np.median(pos)), 0.25)
    idx = np.flatnonzero(dt > thr)
    if not len(idx):
        return t, y
    tg = np.insert(t, idx + 1, t[idx] + dt[idx] / 2)
    yg = np.insert(y.astype(np.float64), idx + 1, np.nan)
    return tg, yg


def gap_threshold(t: np.ndarray) -> float:
    if len(t) < 3:
        return np.inf
    dt = np.diff(t)
    pos = dt[dt > 0]
    if not len(pos):
        return np.inf
    return max(10 * float(np.median(pos)), 0.25)


def short_label(stream, column: str) -> str:
    if stream.root.name == "derived" and stream.loader is None:
        return stream.name if len(stream.columns) == 1 else f"{stream.name}: {column}"
    folder = "/".join(stream.folder.split("/")[-2:]) if stream.folder else ""
    nice = (stream.info.get("col_labels") or {}).get(column)
    if nice:
        head = " ".join(p for p in (folder, stream.info.get("display_name", "")) if p)
        return f"{head} · {nice}"
    name = re.sub(r"\.(csv|tsv|jsonl|db3)$", "", stream.name)
    name = re.sub(r"\.(csv|tsv|jsonl|db3) ", " ", name)
    head = f"{folder}/{name}" if folder else name
    return f"{head}: {column}"


def fmt_value(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e6):
        return f"{v:.3e}"
    return f"{v:.6g}"


def merge_events(events, tol=0.005):
    out = []
    for t, kind, text in sorted(events, key=lambda e: e[0]):
        if out and t - out[-1][0] <= tol:
            pt, pk, ptext = out[-1]
            kinds = pk.split(" / ")
            if kind not in kinds:
                pk = f"{pk} / {kind}"
            out[-1] = (pt, pk, f"{ptext}\n{text}")
        else:
            out.append((t, kind, text))
    return out
