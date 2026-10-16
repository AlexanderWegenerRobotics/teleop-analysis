from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF

TYPES = {
    ".csv": ("CSV", "#2e8b57"),
    ".tsv": ("TSV", "#2e8b57"),
    ".hdf5": ("H5", "#3b6fd4"),
    ".h5": ("H5", "#3b6fd4"),
    ".jsonl": ("JSON", "#d9822b"),
    ".ndjson": ("JSON", "#d9822b"),
    ".json": ("JSON", "#d9822b"),
    ".log": ("LOG", "#6b6b6b"),
    ".txt": ("TXT", "#6b6b6b"),
    ".db3": ("ROS", "#1a9fa8"),
    ".mcap": ("ROS", "#1a9fa8"),
    ".parquet": ("PQ", "#8a4fc7"),
    ".yaml": ("YML", "#9a7b2c"),
    ".yml": ("YML", "#9a7b2c"),
}
VIDEO_COLOR = "#d23c3c"

TAG_W, TAG_H = 30, 16


def _render(kind: str, ext: str, scale: int) -> QPixmap:
    w, h = TAG_W * scale, TAG_H * scale
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)
    if kind == "video":
        text, color = "VID", VIDEO_COLOR
    else:
        text, color = TYPES.get(ext, (ext.lstrip(".")[:4].upper() or "?", "#7a7a7a"))
    r = QRectF(0.5 * scale, 1.5 * scale, w - 1.0 * scale, h - 3.0 * scale)
    rad = 3.0 * scale
    if kind == "events":
        p.setPen(QPen(QColor(color), 1.4 * scale))
        p.setBrush(QColor("white"))
        fg = QColor(color)
    else:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        fg = QColor("white")
    p.drawRoundedRect(r, rad, rad)
    tr = QRectF(r)
    if kind == "video":
        x0, cy, th = r.left() + 3.5 * scale, r.center().y(), 3.2 * scale
        p.setPen(Qt.NoPen)
        p.setBrush(fg)
        p.drawPolygon(QPolygonF([QPointF(x0, cy - th), QPointF(x0, cy + th), QPointF(x0 + 1.6 * th, cy)]))
        tr.setLeft(x0 + 1.6 * th)
    f = QFont()
    f.setBold(True)
    f.setPixelSize(int((8 if len(text) <= 3 else 7) * scale))
    p.setFont(f)
    p.setPen(fg)
    p.drawText(tr, Qt.AlignCenter, text)
    p.end()
    return pm


@lru_cache(maxsize=None)
def file_icon(kind: str, ext: str) -> QIcon:
    icon = QIcon()
    for scale in (1, 2, 3):
        icon.addPixmap(_render(kind, ext.lower(), scale))
    return icon


def icon_for_stream(stream) -> QIcon:
    if stream.video:
        return file_icon("video", "")
    return file_icon("events" if stream.kind == "events" else "signals", Path(stream.rel).suffix)
