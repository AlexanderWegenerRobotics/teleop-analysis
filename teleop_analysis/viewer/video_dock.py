from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QDockWidget, QHBoxLayout, QLabel, QSizePolicy, QSlider, QToolButton,
                               QVBoxLayout, QWidget)

from ..video import FrameReader
from . import theme


class VideoView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 100)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setObjectName("videoView")
        self._img = None

    def show_image(self, img: np.ndarray | None, text: str = "", painter=None):
        if img is None:
            self._img = None
            self.setPixmap(QPixmap())
            self.setText(text or "no frame")
            return
        h, w, _ = img.shape
        self._img = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        if painter is not None:
            qp = QPainter(self._img)
            try:
                painter(qp, w, h)
            finally:
                qp.end()
        self._rescale()

    def _rescale(self):
        if self._img is None:
            return
        self.setPixmap(QPixmap.fromImage(self._img).scaled(self.size(), Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()


class VideoDock(QDockWidget):
    closed = Signal(str)
    timeRequested = Signal(float)

    def __init__(self, key, title, catalog, parent=None, progress=None):
        super().__init__(title, parent)
        self.key = key
        self.catalog = catalog
        self.setObjectName(f"video:{key}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        s = catalog.stream(key)
        self.reader = FrameReader(s.video, progress)
        self.view = VideoView()
        self.info = QLabel("")
        self.info.setObjectName("muted")
        self.info.setContentsMargins(8, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setCursor(Qt.SizeHorCursor)
        self.slider.setToolTip("scrub through the video; moves the time cursor in all plots")
        self.slider.valueChanged.connect(self._slider_moved)
        self.prev_btn = QToolButton()
        theme.set_icon(self.prev_btn, "chevron-left")
        self.prev_btn.setToolTip("Previous frame")
        self.prev_btn.setAutoRepeat(True)
        self.prev_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.next_btn = QToolButton()
        theme.set_icon(self.next_btn, "chevron-right")
        self.next_btn.setToolTip("Next frame")
        self.next_btn.setAutoRepeat(True)
        self.next_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.overlays: dict[str, dict] = {}
        self.overlay_btn = self._make_overlay("intent", "Intent",
                                              "No local.hdf5 (intent log) found next to this video", "target")
        self.gaze_btn = self._make_overlay("gaze", "Gaze", "No gaze log found for this video", "eye-outline")
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 4, 6, 0)
        h.setSpacing(4)
        h.addWidget(self.prev_btn)
        h.addWidget(self.slider, 1)
        h.addWidget(self.next_btn)
        h.addWidget(self.overlay_btn)
        h.addWidget(self.gaze_btn)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(2)
        lay.addWidget(self.view, 1)
        lay.addWidget(bar)
        lay.addWidget(self.info)
        self.setWidget(w)
        self.last_img = None
        self.last_t = None
        self._last_cursor = 0.0
        self.reload_times()
        self._shown = None

    def reload_times(self):
        self.times, self.frames = self.catalog.video_frames(self.key)
        self._shown = None
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(self.times) - 1, 0))
        self.slider.blockSignals(False)

    def _make_overlay(self, name, text, tip, icon):
        btn = QToolButton()
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        theme.set_icon(btn, icon)
        btn.setCheckable(True)
        btn.setEnabled(False)
        btn.setToolTip(tip)
        self.overlays[name] = {"btn": btn, "factory": None, "obj": None}
        btn.toggled.connect(lambda on, n=name: self._overlay_toggled(n, on))
        return btn

    def _active_overlays(self):
        return [o["obj"] for o in self.overlays.values() if o["obj"] is not None and o["btn"].isChecked()]

    def export_image(self):
        if self.last_img is None:
            return None
        q = self.view._img
        if not self._active_overlays() or q is None:
            return self.last_img
        q = q.convertToFormat(QImage.Format_RGB888)
        w, h, bpl = q.width(), q.height(), q.bytesPerLine()
        arr = np.frombuffer(q.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, : 3 * w]
        return arr.reshape(h, w, 3).copy()

    def set_overlay_factory(self, factory, tooltip: str, name: str = "intent"):
        o = self.overlays[name]
        o["factory"] = factory
        o["btn"].setEnabled(factory is not None)
        o["btn"].setToolTip(tooltip)

    def _overlay_toggled(self, name, on):
        o = self.overlays[name]
        if on and o["obj"] is None and o["factory"] is not None:
            try:
                o["obj"] = o["factory"]()
                o["btn"].setToolTip(o["obj"].describe())
            except Exception as e:
                o["btn"].blockSignals(True)
                o["btn"].setChecked(False)
                o["btn"].blockSignals(False)
                o["btn"].setToolTip(f"{name} overlay failed: {e}")
                win = self.window()
                if hasattr(win, "statusBar"):
                    win.statusBar().showMessage(f"{name} overlay: {e}", 8000)
                return
        self._shown = None
        if self.last_t is not None:
            self.show_time(self._last_cursor)

    def refresh_overlay(self):
        for o in self.overlays.values():
            if o["obj"] is not None:
                o["obj"].refresh_times()
        self._shown = None

    def _slider_moved(self, i):
        if 0 <= i < len(self.times):
            self.timeRequested.emit(float(self.times[i]))

    def show_time(self, t: float):
        self._last_cursor = t
        if not len(self.times):
            self.view.show_image(None, "no timestamps")
            return
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        self.slider.blockSignals(True)
        self.slider.setValue(max(i, 0))
        self.slider.blockSignals(False)
        if i < 0:
            self.last_img = None
            self.view.show_image(None, f"video starts at {self.times[0]:.3f} s")
            self.info.setText("")
            self._shown = None
            return
        if i == self._shown:
            return
        self._shown = i
        fi = int(self.frames[i]) if np.isfinite(self.frames[i]) else i
        img = self.reader.frame(fi)
        self.last_img = img
        self.last_t = float(self.times[i])
        lag = (t - self.times[i]) * 1000
        stale = i == len(self.times) - 1 and lag > 1000
        if stale:
            self.last_img = None
        active = self._active_overlays() if img is not None else []
        tf = float(self.times[i])

        def paint_all(qp, w, h):
            for ov in active:
                ov.paint(qp, w, h, tf)
        painter = paint_all if active else None
        self.view.show_image(None if stale else img, "video ended" if stale else "frame unavailable", painter)
        self.info.setText(f"frame {fi}  ·  t = {self.times[i]:.3f} s  ·  cursor − frame = {lag:.0f} ms")

    def closeEvent(self, ev):
        self.reader.close()
        self.closed.emit(self.key)
        super().closeEvent(ev)
