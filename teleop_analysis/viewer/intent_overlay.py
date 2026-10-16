from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen

from ..intent_names import CAND_GOAL, camera_for_video, candidate_names, find_camera_params, phase_names

ARM_COLORS = {"left": QColor("#00c8ff"), "right": QColor("#ff9f1a")}
OBJ_COLOR = QColor("#ffffff")
GOAL_COLOR = QColor("#7dff7a")


def find_local_run(catalog, video_stream):
    folder = video_stream.folder
    root = video_stream.root
    cands = [folder]
    while folder:
        folder = folder.rpartition("/")[0]
        cands.append(folder)
    for f in cands:
        for s in root.streams:
            if s.folder == f and Path(s.rel).name.lower() in ("local.hdf5", "run.hdf5") and s.sub == "frames":
                return s
    return None


class IntentOverlay:
    def __init__(self, catalog, run_stream, video_name: str, extra_param_dirs=None):
        import h5py
        self.catalog = catalog
        self.run_stream = run_stream
        path = run_stream.path
        with h5py.File(path, "r") as f:
            t_ns = np.asarray(f["frames/timestamp_ns"][:], dtype=np.int64)
            self.mask = np.asarray(f["frames/candidate_mask"][:], dtype=bool) if "frames/candidate_mask" in f else None
            self.pos = (np.asarray(f["frames/candidate_world_pos"][:], dtype=float)
                        if "frames/candidate_world_pos" in f else None)
            self.names, self.types, self.names_source = candidate_names(f, path.parent)
            self.phases = phase_names(f)
            self.arms = {}
            for arm in ("left", "right"):
                g = f"modules/intent/{arm}"
                if f"{g}/phase_posterior" in f:
                    self.arms[arm] = (np.asarray(f[f"{g}/phase_posterior"][:], dtype=float),
                                      np.asarray(f[f"{g}/target_posterior"][:], dtype=float)
                                      if f"{g}/target_posterior" in f else None)
        self.t_ns = t_ns
        self.times = catalog.to_seconds(run_stream, t_ns)
        params = find_camera_params(path.parent, extra_param_dirs)
        cam = camera_for_video(params, video_name)
        self.cam = cam if cam and "T_world_cam" in cam and not video_name.lower().startswith(("head", "video_head")) else None

    def refresh_times(self):
        self.times = self.catalog.to_seconds(self.run_stream, self.t_ns)

    def describe(self) -> str:
        cam = "projected into this camera" if self.cam else "no camera model for this video, text only"
        return f"intent overlay from {self.run_stream.rel} (names: {self.names_source}; {cam})"

    def _tick(self, t: float) -> int:
        j = int(np.searchsorted(self.times, t, side="right")) - 1
        return j if 0 <= j < len(self.times) else -1

    def _project(self, p: np.ndarray, w: int, h: int):
        c = self.cam
        T = np.asarray(c["T_world_cam"], dtype=float)
        pc = (np.atleast_2d(p) - T[:3, 3]) @ T[:3, :3]
        z = -pc[:, 2]
        sx = w / float(c.get("width", w))
        sy = h / float(c.get("height", h))
        with np.errstate(all="ignore"):
            u = (c["cx"] + c["fx"] * pc[:, 0] / z) * sx
            v = (c["cy"] - c["fy"] * pc[:, 1] / z) * sy
        bad = (z <= 0.05) | ~np.isfinite(u) | ~np.isfinite(v)
        u[bad] = np.nan
        v[bad] = np.nan
        return u, v

    def paint(self, qp: QPainter, w: int, h: int, t: float):
        j = self._tick(t)
        if j < 0:
            return
        qp.setRenderHint(QPainter.Antialiasing, True)
        base = max(9, int(h / 55))
        font = QFont()
        font.setPixelSize(base)
        bold = QFont(font)
        bold.setBold(True)
        top = {}
        for arm, (ph, tg) in self.arms.items():
            if tg is not None and j < len(tg) and tg.shape[1]:
                k = int(np.argmax(tg[j]))
                top[arm] = (k, float(tg[j, k]))
        if self.cam is not None and self.pos is not None and j < len(self.pos):
            u, v = self._project(self.pos[j], w, h)
            for i in range(len(u)):
                if not np.isfinite(u[i]) or (self.mask is not None and not self.mask[j, i]):
                    continue
                name = self.names[i] if i < len(self.names) and self.names[i] else f"slot {i}"
                goal = i < len(self.types) and self.types[i] == CAND_GOAL
                hits = [(arm, p) for arm, (k, p) in top.items() if k == i and p >= 0.2]
                col = GOAL_COLOR if goal else OBJ_COLOR
                r = base * 0.5
                if goal:
                    qp.setPen(QPen(col, 2))
                    qp.drawLine(int(u[i] - r), int(v[i]), int(u[i] + r), int(v[i]))
                    qp.drawLine(int(u[i]), int(v[i] - r), int(u[i]), int(v[i] + r))
                else:
                    qp.setPen(QPen(col, 2))
                    qp.setBrush(Qt.NoBrush)
                    qp.drawEllipse(QRectF(u[i] - r, v[i] - r, 2 * r, 2 * r))
                for n, (arm, p) in enumerate(hits):
                    rr = r * (2.0 + 0.8 * n)
                    qp.setPen(QPen(ARM_COLORS[arm], 3))
                    qp.drawEllipse(QRectF(u[i] - rr, v[i] - rr, 2 * rr, 2 * rr))
                text = name + "".join(f"  {arm[0].upper()} {p:.0%}" for arm, p in hits)
                self._label(qp, bold if hits else font, text, u[i] + r * 2.2, v[i] - base * 0.6,
                            ARM_COLORS[hits[0][0]] if hits else col)
        y = base * 0.6
        for arm, (ph, tg) in self.arms.items():
            if j >= len(ph):
                continue
            k = int(np.argmax(ph[j]))
            pname = self.phases[k] if k < len(self.phases) else str(k)
            text = f"{arm[0].upper()}  {pname} {ph[j, k]:.0%}"
            if arm in top:
                ti, tp = top[arm]
                tn = self.names[ti] if ti < len(self.names) and self.names[ti] else f"slot {ti}"
                text += f"   →  {tn} {tp:.0%}"
            self._label(qp, bold, text, base * 0.6, y, ARM_COLORS[arm])
            y += base * 1.6

    @staticmethod
    def _label(qp: QPainter, font: QFont, text: str, x: float, y: float, color: QColor):
        qp.setFont(font)
        fm = QFontMetrics(font)
        rect = QRectF(x, y, fm.horizontalAdvance(text) + 8, fm.height() + 2)
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor(0, 0, 0, 190))
        qp.drawRect(rect)
        qp.setPen(QPen(color))
        qp.drawText(rect.adjusted(4, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text)


GAZE_COLOR = QColor("#ff3b3b")
GAZE_TRAIL_S = 0.4
GAZE_HOLD_S = 0.25


def _is_head_video(name: str) -> bool:
    return name.lower().startswith(("head", "video_head"))


def find_gaze_sources(catalog, video_stream) -> list[tuple[str, object]]:
    root = video_stream.root
    if Path(video_stream.rel).name.lower().endswith((".jsonl", ".ndjson")):
        prim = next((s for s in root.streams if s.rel == video_stream.rel and s.sub is None), None)
        if prim is not None and "gaze_u" in prim.columns and "gaze_v" in prim.columns:
            return [("interface", prim)]
        return []
    folder = video_stream.folder
    cands = [folder]
    while folder:
        folder = folder.rpartition("/")[0]
        cands.append(folder)
    out = []
    for f in cands:
        for s in root.streams:
            if s.folder == f and "gaze_px_x" in s.columns and "gaze_px_y" in s.columns:
                out.append(("camera", s))
    if not _is_head_video(Path(video_stream.rel).name):
        for f in cands:
            for s in root.streams:
                if (s.sub is None and (s.folder == f or s.folder.startswith(f + "/") if f else True)
                        and Path(s.rel).name.lower().endswith((".jsonl", ".ndjson"))
                        and "gaze_u" in s.columns and "gaze_v" in s.columns):
                    out.append(("interface", s))
            if out and out[-1][0] == "interface":
                break
    return out


def find_gaze_source(catalog, video_stream):
    c = find_gaze_sources(catalog, video_stream)
    return c[0] if c else None


def make_gaze_overlay(catalog, sources, video_name: str, folder: Path):
    errors = []
    for mode, s in sources:
        try:
            ov = GazeOverlay(catalog, mode, s, video_name, folder)
        except Exception as e:
            errors.append(f"{Path(s.rel).name}: {e}")
            continue
        if len(ov.t):
            return ov
        errors.append(f"{Path(s.rel).name}: no valid gaze samples")
    raise ValueError("; ".join(errors) or "no gaze log found")


class GazeOverlay:
    def __init__(self, catalog, mode: str, stream, video_name: str, folder: Path):
        self.catalog = catalog
        self.mode = mode
        self.stream = stream
        if mode == "interface":
            cols = ("gaze_u", "gaze_v")
            self.cam = None
        else:
            cols = ("gaze_px_x", "gaze_px_y")
            self.cam = camera_for_video(find_camera_params(folder), video_name)
            if self.cam is None:
                raise ValueError("no camera model (camera_params.json) for this video")
        self.cols = cols
        self.valid_col = "gaze_valid" if "gaze_valid" in stream.columns else None
        self.refresh_times()

    def refresh_times(self):
        t, u = self.catalog.get(self.stream.signal_key(self.cols[0]))
        _, v = self.catalog.get(self.stream.signal_key(self.cols[1]))
        ok = np.isfinite(u) & np.isfinite(v)
        if self.valid_col:
            _, val = self.catalog.get(self.stream.signal_key(self.valid_col))
            ok &= np.nan_to_num(val) > 0.5
        self.t, self.u, self.v = t[ok], u[ok], v[ok]

    def describe(self) -> str:
        src = f"{self.stream.rel} (gaze in the streamed image)" if self.mode == "interface" else \
            f"{self.stream.rel} (gaze rays through the camera model)"
        return f"gaze from {src}; dot = latest sample, trail = last {GAZE_TRAIL_S:.1f} s"

    def _px(self, u, v, w, h):
        if self.mode == "interface":
            return u * w, v * h
        c = self.cam
        sx = w / float(c.get("width", w))
        sy = h / float(c.get("height", h))
        return (c["cx"] + c["fx"] * u) * sx, (c["cy"] + c["fy"] * v) * sy

    def paint(self, qp: QPainter, w: int, h: int, t: float):
        j = int(np.searchsorted(self.t, t, side="right")) - 1
        if j < 0 or t - self.t[j] > GAZE_HOLD_S:
            return
        i0 = int(np.searchsorted(self.t, t - GAZE_TRAIL_S, side="left"))
        qp.setRenderHint(QPainter.Antialiasing, True)
        n = j - i0 + 1
        r0 = max(3.0, h / 160)
        qp.setPen(Qt.NoPen)
        for k in range(i0, j):
            x, y = self._px(self.u[k], self.v[k], w, h)
            a = int(40 + 140 * (k - i0 + 1) / max(n, 1))
            c = QColor(GAZE_COLOR)
            c.setAlpha(a)
            qp.setBrush(c)
            qp.drawEllipse(QRectF(x - r0, y - r0, 2 * r0, 2 * r0))
        x, y = self._px(self.u[j], self.v[j], w, h)
        R = max(10.0, h / 30)
        qp.setBrush(Qt.NoBrush)
        qp.setPen(QPen(QColor(0, 0, 0, 160), 4))
        qp.drawEllipse(QRectF(x - R, y - R, 2 * R, 2 * R))
        qp.setPen(QPen(GAZE_COLOR, 2))
        qp.drawEllipse(QRectF(x - R, y - R, 2 * R, 2 * R))
        qp.setBrush(GAZE_COLOR)
        qp.setPen(Qt.NoPen)
        qp.drawEllipse(QRectF(x - r0, y - r0, 2 * r0, 2 * r0))
