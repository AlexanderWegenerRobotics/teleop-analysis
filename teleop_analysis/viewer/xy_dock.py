"""XY view: one plotted signal against another instead of against time."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QDockWidget, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from . import theme
from .common import gap_threshold


class XYDock(QDockWidget):
    """y(t) over x(t), both resampled onto the x signal's timestamps. Follows the visible time window and cursor A."""

    def __init__(self, area, cx, cy, parent=None):
        super().__init__("XY", parent)
        self.setObjectName(f"xy:{id(self)}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.area = area
        self.cx, self.cy = cx, cy
        self.plot = pg.PlotWidget(background=theme.t("plot_bg"))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.line = self.plot.plot([], [], pen=pg.mkPen(theme.t("muted"), width=0.8))
        self.dots = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(cy.color))
        self.plot.addItem(self.dots)
        self.cursor = pg.ScatterPlotItem(size=11, pen=pg.mkPen(theme.t("plot_bg"), width=1.5),
                                         brush=pg.mkBrush(theme.t("cursor_a")))
        self.plot.addItem(self.cursor)
        self.window_only = QCheckBox("visible time window only")
        self.window_only.setChecked(True)
        self.window_only.toggled.connect(self.refresh)
        self.trail = QCheckBox("connect points")
        self.trail.setChecked(True)
        self.trail.toggled.connect(self.refresh)
        swap = QToolButton()
        swap.setText("Swap x/y")
        swap.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        theme.set_icon(swap, "swap-horizontal")
        swap.clicked.connect(self._swap)
        self.info = QLabel("")
        self.info.setObjectName("muted")
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        for w in (self.window_only, self.trail, swap):
            h.addWidget(w)
        h.addStretch(1)
        h.addWidget(self.info)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.plot, 1)
        v.addWidget(bar)
        self.setWidget(body)
        area.cursorMoved.connect(self._cursor)
        area.xRangeChanged.connect(lambda *_: self.refresh())
        area.curvesChanged.connect(self._check_alive)
        self._labels()
        self.refresh()

    def _labels(self):
        self.setWindowTitle(f"XY: {self.cy.label}  vs  {self.cx.label}")
        self.plot.setLabel("bottom", self.cx.label)
        self.plot.setLabel("left", self.cy.label)
        self.dots.setBrush(pg.mkBrush(self.cy.color))

    def _swap(self):
        self.cx, self.cy = self.cy, self.cx
        self._labels()
        self.refresh()

    def _check_alive(self):
        alive = {id(c) for _, c in self.area.all_curves()}
        if id(self.cx) not in alive or id(self.cy) not in alive:
            self.close()

    def _samples(self):
        """x and y on the x signal's timestamps inside the chosen window; gaps in either signal are dropped."""
        t, x = self.cx.t, self.cx.y
        if self.window_only.isChecked() and self.area.x_range() is not None:
            a, b = self.area.x_range()
            m = (t >= a) & (t <= b)
            t, x = t[m], x[m]
        if not len(t) or not len(self.cy.t):
            return t, x, np.zeros(0)
        j = np.searchsorted(self.cy.t, t, side="right") - 1
        ok = (j >= 0) & (t - self.cy.t[np.clip(j, 0, len(self.cy.t) - 1)] <= gap_threshold(self.cy.t))
        y = np.full(len(t), np.nan)
        y[ok] = np.interp(t[ok], self.cy.t, self.cy.y)
        keep = ok & np.isfinite(x) & np.isfinite(y)
        return t[keep], x[keep], y[keep]

    def refresh(self):
        t, x, y = self._samples()
        self.line.setData(x, y) if self.trail.isChecked() else self.line.setData([], [])
        self.dots.setData(x, y)
        self.info.setText(f"{len(x)} points")
        self._cursor(self.area.cursor_t)

    def _cursor(self, tc):
        _, x = self.cx.value_at(tc)
        _, y = self.cy.value_at(tc)
        if x is None or y is None:
            self.cursor.setData([], [])
        else:
            self.cursor.setData([x], [y])
