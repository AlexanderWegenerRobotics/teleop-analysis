from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtWidgets import QInputDialog, QLabel, QSplitter, QVBoxLayout, QWidget

from . import theme
from .common import MOVE_MIME, PALETTE, event_color, fmt_value, gap_threshold, merge_events, mime_keys, with_gaps

pg.setConfigOptions(antialias=False)
HOVER_PX = 10


class TimeViewBox(pg.ViewBox):
    def __init__(self, panel_ref):
        super().__init__()
        self._panel = panel_ref
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, ev, axis=None):
        if axis is None and not (ev.modifiers() & Qt.ShiftModifier):
            axis = 0
        super().wheelEvent(ev, axis)

    def mouseClickEvent(self, ev):
        p = self._panel()
        if ev.button() == Qt.RightButton and p is not None and p.hover_curve is not None:
            ev.accept()
            p.curveMenuRequested.emit(p, p.hover_curve, ev.screenPos().toPoint())
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if axis is None and ev.button() == Qt.RightButton:
            ev.accept()
            if ev.isStart():
                self.setCursor(Qt.CrossCursor)
            p0, p1 = ev.buttonDownPos(), ev.pos()
            if ev.isFinish():
                self.rbScaleBox.hide()
                self.setCursor(Qt.OpenHandCursor)
                if (p1 - p0).manhattanLength() > 6:
                    rect = self.childGroup.mapRectFromParent(QRectF(p0, p1).normalized())
                    panel = self._panel()
                    if panel is not None:
                        panel.area.push_zoom()
                    self.disableAutoRange(axis=pg.ViewBox.YAxis)
                    self.setRange(rect, padding=0)
            else:
                self.updateScaleBox(p0, p1)
            return
        if ev.isStart():
            self.setCursor(Qt.ClosedHandCursor if ev.button() == Qt.LeftButton else Qt.SizeAllCursor)
        super().mouseDragEvent(ev, axis)
        if ev.isFinish():
            self.setCursor(Qt.OpenHandCursor)


class Curve:
    def __init__(self, key, label, color, item, t, y):
        self.key = key
        self.label = label
        self.color = color
        self.item = item
        self.t = t
        self.y = y
        self.width = 1.3
        self.style = "solid"
        self.selected = False
        self.gap = gap_threshold(t)

    def value_at(self, x):
        if not len(self.t):
            return None, None
        j = int(np.searchsorted(self.t, x, side="right")) - 1
        if j < 0 or x - self.t[j] > self.gap:
            return None, None
        return float(self.t[j]), float(self.y[j])


class PlotPanel(pg.PlotWidget):
    dropped = Signal(object, list)
    moved = Signal(object, list)
    clicked = Signal(object, float, object)
    removeRequested = Signal(object)
    changed = Signal()
    curveMenuRequested = Signal(object, object, object)
    selectToggled = Signal(object, object)

    def __init__(self, area):
        import weakref
        ref = weakref.ref(self)
        self.vb = TimeViewBox(lambda: ref())
        super().__init__(viewBox=self.vb)
        self.area = area
        self.curves: list[Curve] = []
        self.event_items = []
        self.y_label = ""
        self.hover_curve: Curve | None = None
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.legend = self.addLegend(offset=(5, 5), labelTextSize="8pt")
        self.legend.setCursor(Qt.SizeAllCursor)
        self.legend.setToolTip("drag to move the legend")
        self.vb.enableAutoRange(axis="y")
        self.vb.setAutoVisible(y=True)
        self.setMinimumHeight(90)
        self.getAxis("left").setCursor(Qt.SizeVerCursor)
        self.getAxis("left").setToolTip("wheel / drag here: zoom / pan y only")
        self.getAxis("bottom").setCursor(Qt.SizeHorCursor)

        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(theme.t("cursor_a"), width=1.5))
        self.cursor.setZValue(1000)
        self.cursor.setCursor(Qt.SizeHorCursor)
        self.addItem(self.cursor, ignoreBounds=True)
        self.cursor.sigPositionChanged.connect(lambda line: self.area.set_cursor(line.value(), source=self))

        self.cursor_b = pg.InfiniteLine(angle=90, movable=True,
                                        pen=pg.mkPen(theme.t("cursor_b"), width=1.5, style=Qt.DashLine))
        self.cursor_b.setZValue(999)
        self.cursor_b.setCursor(Qt.SizeHorCursor)
        self.cursor_b.setVisible(False)
        self.addItem(self.cursor_b, ignoreBounds=True)
        self.cursor_b.sigPositionChanged.connect(lambda line: self.area.set_cursor_b(line.value(), source=self))
        self.b_label = None

        self.ycur = []
        for i in range(2):
            ln = pg.InfiniteLine(angle=0, movable=True)
            ln.setCursor(Qt.SizeVerCursor)
            ln.setVisible(False)
            ln.setZValue(998)
            ln.label = pg.InfLineLabel(ln, text="", position=0.98, anchors=[(1, 1), (1, 1)],
                                       color=theme.t("ycursor"))
            ln.sigPositionChanged.connect(lambda *_: self._update_ycur_labels())
            self.addItem(ln, ignoreBounds=True)
            self.ycur.append(ln)

        self.hover_text = pg.TextItem(anchor=(0, 1))
        self.hover_text.setZValue(2000)
        self.hover_text.setVisible(False)
        self.addItem(self.hover_text, ignoreBounds=True)
        self.hover_dot = pg.ScatterPlotItem(size=9)
        self.hover_dot.setZValue(1999)
        self.addItem(self.hover_dot, ignoreBounds=True)
        self._move_proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

        self.scene().sigMouseClicked.connect(self._scene_click)
        menu = self.vb.menu
        menu.addSeparator()
        self.remove_menu = menu.addMenu("Remove signal")
        self.math_menu = menu.addMenu("Math on selected signals")
        self.ycur_action = menu.addAction("Y cursors (Δy)")
        self.ycur_action.setCheckable(True)
        self.ycur_action.toggled.connect(self.show_ycursors)
        menu.addAction("Set y label…", self._set_ylabel)
        menu.addAction("Fit y", self.fit_y)
        menu.addAction("Remove panel", lambda: self.removeRequested.emit(self))
        menu.aboutToShow.connect(self._fill_menus)
        self._active = False
        self.apply_theme()

    def apply_theme(self):
        c = theme.t
        self.setBackground(c("plot_bg"))
        for name in ("left", "bottom"):
            ax = self.getAxis(name)
            ax.setPen(pg.mkPen(c("border")))
            ax.setTextPen(pg.mkPen(c("plot_fg")))
        self.showGrid(x=True, y=True, alpha=0.18 if theme.name() == "dark" else 0.25)
        self.legend.setBrush(pg.mkBrush(*c("overlay")))
        self.legend.setPen(pg.mkPen(c("border")))
        self.legend.setLabelTextColor(c("text"))
        for _, label in self.legend.items:
            label.setText(label.text, color=c("text"))
        self.cursor.setPen(pg.mkPen(c("cursor_a"), width=1.6))
        self.cursor.setHoverPen(pg.mkPen(c("cursor_a"), width=3))
        self.cursor_b.setPen(pg.mkPen(c("cursor_b"), width=1.6, style=Qt.DashLine))
        self.cursor_b.setHoverPen(pg.mkPen(c("cursor_b"), width=3))
        for ln in self.ycur:
            ln.setPen(pg.mkPen(c("ycursor"), width=1.2, style=Qt.DashLine))
            ln.setHoverPen(pg.mkPen(c("ycursor"), width=3))
            ln.label.setColor(c("ycursor"))
            ln.label.fill = pg.mkBrush(*c("overlay"))
        if self.b_label is not None:
            self.b_label.setColor(c("cursor_b"))
            self.b_label.fill = pg.mkBrush(*c("overlay"))
        self.hover_text.fill = pg.mkBrush(*c("overlay"))
        self.hover_text.border = pg.mkPen(c("border"))
        self.hover_dot.setPen(pg.mkPen(c("plot_bg"), width=1.5))
        self.set_active(self._active)
        self.update()

    def _fill_menus(self):
        self.remove_menu.clear()
        for c in list(self.curves):
            self.remove_menu.addAction(c.label, lambda c=c: self.remove_curve(c))
        if self.curves:
            self.remove_menu.addSeparator()
            self.remove_menu.addAction("All", self.clear_curves)
        self.math_menu.clear()
        hook = getattr(self.area, "math_menu_hook", None)
        if hook:
            hook(self.math_menu)
        self.math_menu.setEnabled(not self.math_menu.isEmpty())

    def _set_ylabel(self):
        text, ok = QInputDialog.getText(self, "Y label", "Label (e.g. force [N]):", text=self.y_label)
        if ok:
            self.set_ylabel(text)

    def set_ylabel(self, text):
        self.y_label = text
        if text:
            self.setLabel("left", text)
        else:
            self.getAxis("left").setLabel(None)
            self.getAxis("left").showLabel(False)
        self.changed.emit()

    def fit_y(self):
        self.vb.enableAutoRange(axis="y")
        self.vb.setAutoVisible(y=True)

    def set_active(self, on):
        self._active = on
        self.vb.setBorder(pg.mkPen(theme.t("accent"), width=2) if on else None)

    def show_ycursors(self, on):
        if self.ycur_action.isChecked() != on:
            self.ycur_action.blockSignals(True)
            self.ycur_action.setChecked(on)
            self.ycur_action.blockSignals(False)
        if on:
            lo, hi = self.vb.viewRange()[1]
            self.ycur[0].setValue(lo + (hi - lo) / 3)
            self.ycur[1].setValue(lo + 2 * (hi - lo) / 3)
        for ln in self.ycur:
            ln.setVisible(on)
        self._update_ycur_labels()

    def _update_ycur_labels(self):
        y1, y2 = self.ycur[0].value(), self.ycur[1].value()
        self.ycur[0].label.setFormat(f"y₁ = {fmt_value(y1)}")
        self.ycur[1].label.setFormat(f"y₂ = {fmt_value(y2)}   Δy = {fmt_value(y2 - y1)}")

    def ycursor_values(self):
        if not self.ycur[0].isVisible():
            return None
        return self.ycur[0].value(), self.ycur[1].value()

    def _scene_click(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        if not self.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        if ev.double():
            self.area.push_zoom()
            self.area.fit_all(push=False)
            return
        x = self.vb.mapSceneToView(ev.scenePos()).x()
        mods = ev.modifiers()
        if mods & Qt.ControlModifier:
            if self.hover_curve is not None:
                self.selectToggled.emit(self, self.hover_curve)
            return
        self.clicked.emit(self, x, mods)

    def _mouse_moved(self, args):
        pos = args[0]
        if not self.vb.sceneBoundingRect().contains(pos) or not self.curves:
            self._hide_hover()
            return
        x = self.vb.mapSceneToView(pos).x()
        px = self.vb.viewPixelSize()[0] or 1e-9
        best = None
        for c in self.curves:
            if not c.item.isVisible() or not len(c.t):
                continue
            lo = int(np.searchsorted(c.t, x - 4 * px, side="left"))
            hi = int(np.searchsorted(c.t, x + 4 * px, side="right"))
            lo = max(lo - 1, 0)
            hi = min(max(hi, lo + 2), len(c.t))
            seg = c.y[lo:hi]
            if not np.isfinite(seg).any():
                continue
            j = lo + int(np.argmin(np.abs(c.t[lo:hi] - x)))
            if not np.isfinite(c.y[j]) or abs(c.t[j] - x) > max(c.gap, 6 * px):
                continue
            ymin = self.vb.mapViewToScene(QPointF(x, float(np.nanmin(seg)))).y()
            ymax = self.vb.mapViewToScene(QPointF(x, float(np.nanmax(seg)))).y()
            top, bot = min(ymin, ymax), max(ymin, ymax)
            d = 0.0 if top <= pos.y() <= bot else min(abs(pos.y() - top), abs(pos.y() - bot))
            if d <= HOVER_PX and (best is None or d < best[0]):
                best = (d, c, j)
        if best is None:
            self._hide_hover()
            return
        _, c, j = best
        self.hover_curve = c
        t, y = float(c.t[j]), float(c.y[j])
        self.hover_dot.setData([t], [y], brush=pg.mkBrush(c.color))
        fg = theme.t("text")
        self.hover_text.setHtml(f"<div style='font-size:9pt; color:{fg}'><b style='color:{c.color}'>{c.label}</b><br>"
                                f"t = {t:.4f} s<br>y = {fmt_value(y)}</div>")
        vr = self.vb.viewRange()
        right = (t - vr[0][0]) > 0.6 * (vr[0][1] - vr[0][0])
        top = (y - vr[1][0]) > 0.7 * (vr[1][1] - vr[1][0])
        self.hover_text.setAnchor((1 if right else 0, 0 if top else 1))
        self.hover_text.setPos(t, y)
        self.hover_text.setVisible(True)
        self.vb.setCursor(Qt.PointingHandCursor)
        self.setToolTip("")

    def _hide_hover(self):
        if self.hover_curve is not None or self.hover_text.isVisible():
            self.hover_curve = None
            self.hover_text.setVisible(False)
            self.hover_dot.setData([], [])
            self.vb.setCursor(Qt.OpenHandCursor)

    def leaveEvent(self, ev):
        self._hide_hover()
        super().leaveEvent(ev)

    def dragEnterEvent(self, ev):
        if mime_keys(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        if mime_keys(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        keys = mime_keys(ev.mimeData())
        if keys:
            ev.acceptProposedAction()
            if ev.mimeData().hasFormat(MOVE_MIME):
                self.moved.emit(self, keys)
            else:
                self.dropped.emit(self, keys)

    def has(self, key):
        return any(c.key == key for c in self.curves)

    def _pen(self, c):
        w = c.width * (2.3 if c.selected else 1.0)
        style = {"dash": Qt.DashLine, "dot": Qt.DotLine, "dashdot": Qt.DashDotLine}.get(c.style, Qt.SolidLine)
        return pg.mkPen(c.color, width=w, style=style)

    def set_style(self, c, style):
        c.style = style
        self.restyle(c)
        self.changed.emit()

    def add_curve(self, key, t, y, label, color=None, width=None, style=None):
        if self.has(key):
            return None
        if color is None:
            used = {c.color for c in self.curves}
            color = next((c for c in PALETTE if c not in used), PALETTE[len(self.curves) % len(PALETTE)])
        tg, yg = with_gaps(t, y)
        kw = dict(pen=pg.mkPen(color, width=1.3), name=label, connect="finite")
        if len(t) < 300:
            kw.update(symbol="o", symbolSize=4, symbolPen=None, symbolBrush=color)
        item = pg.PlotDataItem(tg, yg, **kw)
        self.addItem(item)
        item.setClipToView(True)
        item.setDownsampling(auto=True, method="peak")
        c = Curve(key, label, color, item, t, y)
        if width is not None:
            c.width = float(width)
        if style:
            c.style = style
        if width is not None or style:
            self.restyle(c)
        self.curves.append(c)
        self.changed.emit()
        return c

    def refresh_curve(self, c, t, y):
        c.t, c.y = t, y
        c.gap = gap_threshold(t)
        tg, yg = with_gaps(t, y)
        c.item.setData(tg, yg)

    def restyle(self, c):
        c.item.setPen(self._pen(c))
        if c.item.opts.get("symbol"):
            c.item.setSymbolBrush(c.color)

    def rename_curve(self, c, label):
        c.label = label
        self.legend.removeItem(c.item)
        c.item.opts["name"] = label
        self.legend.addItem(c.item, label)
        self.changed.emit()

    def recolor_curve(self, c, color):
        c.color = color
        self.restyle(c)
        self.changed.emit()

    def set_width(self, c, w):
        c.width = w
        self.restyle(c)

    def remove_curve(self, c):
        if c is self.hover_curve:
            self._hide_hover()
        self.removeItem(c.item)
        self.legend.removeItem(c.item)
        self.curves.remove(c)
        self.area.forget(self, c)
        self.changed.emit()

    def clear_curves(self):
        for c in list(self.curves):
            self.remove_curve(c)

    def set_events(self, events, labels=True):
        for it in self.event_items:
            self.removeItem(it)
        self.event_items = []
        for t, kind, text in merge_events(events):
            col = event_color(kind.split(" / ")[0])
            line = pg.InfiniteLine(pos=t, angle=90, movable=False, pen=pg.mkPen(col, width=1, style=Qt.DashLine))
            if labels:
                line.label = pg.InfLineLabel(line, text=kind, position=0.97, anchors=[(0, 0), (0, 0)],
                                             color=col, fill=pg.mkBrush(*theme.t("overlay")))
                line.label.setFont(pg.QtGui.QFont("", 8))
            line.setToolTip(text)
            line.setZValue(-10)
            self.addItem(line, ignoreBounds=True)
            self.event_items.append(line)

    def set_b_label(self, on, text=""):
        if on and self.b_label is None:
            self.b_label = pg.InfLineLabel(self.cursor_b, text="", position=0.88, anchors=[(0, 0), (0, 0)],
                                           color=theme.t("cursor_b"), fill=pg.mkBrush(*theme.t("overlay")))
        if self.b_label is not None:
            self.b_label.setVisible(on)
            if on:
                self.b_label.setFormat(text)

    def visible_curves(self):
        return [c for c in self.curves if c.item.isVisible()]


class DropStrip(QLabel):
    dropped = Signal(list)
    moved = Signal(list)

    def __init__(self):
        super().__init__("Drop signals here for a new panel")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(34)
        self.setObjectName("dropStrip")
        self.setMinimumHeight(40)

    def _hot(self, on):
        self.setProperty("hot", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, ev):
        if mime_keys(ev.mimeData()):
            ev.acceptProposedAction()
            self._hot(True)

    def dragLeaveEvent(self, ev):
        self._hot(False)

    def dropEvent(self, ev):
        self._hot(False)
        keys = mime_keys(ev.mimeData())
        if keys:
            ev.acceptProposedAction()
            if ev.mimeData().hasFormat(MOVE_MIME):
                self.moved.emit(keys)
            else:
                self.dropped.emit(keys)




class PlotArea(QWidget):
    cursorMoved = Signal(float)
    cursorBMoved = Signal(float)
    xRangeChanged = Signal(float, float)
    curvesChanged = Signal()
    selectionChanged = Signal()
    loadRequested = Signal(object, list)
    moveRequested = Signal(object, list)
    curveMenuRequested = Signal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.strip = DropStrip()
        self.strip.dropped.connect(lambda keys: self.loadRequested.emit(None, keys))
        self.strip.moved.connect(lambda keys: self.moveRequested.emit(None, keys))
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(2)
        self.root = self._splitter("v")
        self.lay.addWidget(self.root, 1)
        self.lay.addWidget(self.strip)
        self.panels: list[PlotPanel] = []
        self.active: PlotPanel | None = None
        self.cursor_t = 0.0
        self.cursor_b_t = 0.0
        self.b_enabled = False
        self.selected: list[tuple[PlotPanel, Curve]] = []
        self._sync = False
        self._sync_b = False
        self.events: list = []
        self.event_labels = True
        self.math_menu_hook = None
        self.zoom_stack: list = []

    def _splitter(self, orient):
        sp = QSplitter(Qt.Vertical if orient == "v" else Qt.Horizontal)
        sp.setChildrenCollapsible(False)
        sp.setHandleWidth(7)
        return sp

    def _new_panel(self) -> PlotPanel:
        p = PlotPanel(self)
        p.dropped.connect(lambda panel, keys: self.loadRequested.emit(panel, keys))
        p.moved.connect(lambda panel, keys: self.moveRequested.emit(panel, keys))
        p.clicked.connect(self._clicked)
        p.removeRequested.connect(self.remove_panel)
        p.changed.connect(self.curvesChanged.emit)
        p.curveMenuRequested.connect(self.curveMenuRequested.emit)
        p.selectToggled.connect(self.toggle_select)
        for line, val in ((p.cursor, self.cursor_t), (p.cursor_b, self.cursor_b_t)):
            line.blockSignals(True)
            line.setValue(val)
            line.blockSignals(False)
        p.cursor_b.setVisible(self.b_enabled)
        return p

    def _collect(self, w):
        if isinstance(w, PlotPanel):
            return [w]
        if isinstance(w, QSplitter):
            out = []
            for i in range(w.count()):
                out += self._collect(w.widget(i))
            return out
        return []

    def _after_change(self, new_panels=()):
        r = self.x_range() if self.panels else None
        self.panels = self._collect(self.root)
        for p in new_panels:
            p.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
        self._relink()
        if r is not None and self.panels:
            self._xsyncing = True
            try:
                for p in self.panels:
                    p.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
                    p.vb.setXRange(r[0], r[1], padding=0)
            finally:
                self._xsyncing = False
        self._apply_events()
        if new_panels:
            self.set_active(new_panels[-1])
        elif self.active not in self.panels:
            self.active = None
            if self.panels:
                self.set_active(self.panels[-1])
        else:
            self.set_active(self.active)
        self._update_b_label()
        self.selectionChanged.emit()
        self.curvesChanged.emit()

    def add_panel(self) -> PlotPanel:
        p = self._new_panel()
        if self.root.orientation() != Qt.Vertical:
            old = self.root
            self.lay.removeWidget(old)
            self.root = self._splitter("v")
            self.lay.insertWidget(0, self.root, 1)
            self.root.addWidget(old)
        self.root.addWidget(p)
        self._after_change([p])
        return p

    def _collapse(self, sp):
        if not isinstance(sp, QSplitter) or sp is self.root:
            return
        if sp.count() == 0:
            g = sp.parentWidget()
            sp.setParent(None)
            sp.deleteLater()
            self._collapse(g)
        elif sp.count() == 1:
            child = sp.widget(0)
            g = sp.parentWidget()
            if isinstance(g, QSplitter):
                g.insertWidget(g.indexOf(sp), child)
                sp.setParent(None)
                sp.deleteLater()

    def remove_panel(self, p):
        if p not in self.panels:
            return
        self.selected = [(q, c) for q, c in self.selected if q is not p]
        parent = p.parentWidget()
        p.setParent(None)
        p.deleteLater()
        self._collapse(parent)
        self._after_change()

    def clear(self):
        for p in list(self.panels):
            self.remove_panel(p)

    def _build(self, spec, take, sizes_todo):
        if not isinstance(spec, (list, tuple)):
            return take(spec)
        orient, children = spec[0], spec[1]
        sp = self._splitter(orient)
        for ch in children:
            sp.addWidget(self._build(ch, take, sizes_todo))
        if len(spec) > 2 and spec[2]:
            sizes_todo.append((sp, list(spec[2])))
        return sp

    def apply_layout(self, spec, keep=True):
        old = list(self.panels) if keep else []
        if not keep:
            self.selected = []
            for p in self.panels:
                p.setParent(None)
                p.deleteLater()
        for p in old:
            p.setParent(None)
        self.lay.removeWidget(self.root)
        self.root.setParent(None)
        self.root.deleteLater()
        pool = iter(old)
        created = []

        def take(_):
            try:
                return next(pool)
            except StopIteration:
                p = self._new_panel()
                created.append(p)
                return p
        sizes_todo = []
        root = self._build(spec, take, sizes_todo)
        if not isinstance(root, QSplitter):
            sp = self._splitter("v")
            sp.addWidget(root)
            root = sp
        extra = list(pool)
        if extra:
            if root.orientation() != Qt.Vertical:
                sp = self._splitter("v")
                sp.addWidget(root)
                root = sp
            for p in extra:
                root.addWidget(p)
        self.root = root
        self.lay.insertWidget(0, self.root, 1)

        def apply_sizes():
            for sp, w in sizes_todo:
                tot = sum(w) or 1
                full = sum(sp.sizes()) or 1000
                sp.setSizes([int(full * x / tot) for x in w])
        apply_sizes()
        QTimer.singleShot(0, apply_sizes)
        self._after_change(created)

    def layout_spec(self):
        idx = {id(p): i for i, p in enumerate(self.panels)}

        def walk(w):
            if isinstance(w, PlotPanel):
                return idx.get(id(w), 0)
            return ["v" if w.orientation() == Qt.Vertical else "h",
                    [walk(w.widget(i)) for i in range(w.count())], list(w.sizes())]
        return walk(self.root)

    def _edge(self, p, last):
        w = p
        while w is not self.root:
            sp = w.parentWidget()
            if not isinstance(sp, QSplitter):
                break
            if sp.orientation() == Qt.Vertical:
                i = sp.indexOf(w)
                if (last and i != sp.count() - 1) or (not last and i != 0):
                    return False
            w = sp
        return True

    def _relink(self):
        if not self.panels:
            return
        for p in self.panels:
            p.setXLink(None)
            if not getattr(p, "_xsync", False):
                p.vb.sigXRangeChanged.connect(self._x_changed)
                p._xsync = True
        for i, p in enumerate(self.panels):
            bottom = self._edge(p, True)
            p.getAxis("bottom").setStyle(showValues=bottom)
            if bottom:
                p.setLabel("bottom", "time [s]")
            else:
                p.getAxis("bottom").showLabel(False)

    def _x_changed(self, vb, rng):
        if getattr(self, "_xsyncing", False):
            return
        self._xsyncing = True
        try:
            for p in self.panels:
                if p.vb is not vb:
                    p.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
                    p.vb.setXRange(rng[0], rng[1], padding=0)
        finally:
            self._xsyncing = False
        self.xRangeChanged.emit(float(rng[0]), float(rng[1]))

    def set_active(self, p):
        self.active = p
        for q in self.panels:
            q.set_active(q is p and len(self.panels) > 1)

    def target_panel(self) -> PlotPanel:
        if self.active is not None and self.active in self.panels:
            return self.active
        if self.panels:
            return self.panels[-1]
        return self.add_panel()

    def _clicked(self, panel, x, mods):
        self.set_active(panel)
        if mods & Qt.ShiftModifier:
            self.enable_b(True, x)
        else:
            self.set_cursor(x)

    def set_cursor(self, t, source=None):
        if self._sync:
            return
        self._sync = True
        try:
            self.cursor_t = float(t)
            for p in self.panels:
                if p is not source:
                    p.cursor.setValue(self.cursor_t)
        finally:
            self._sync = False
        self._update_b_label()
        self.cursorMoved.emit(self.cursor_t)

    def enable_b(self, on, t=None):
        self.b_enabled = on
        if on:
            if t is None:
                r = self.x_range()
                w = (r[1] - r[0]) if r else 1.0
                t = self.cursor_t + 0.1 * w
            self.set_cursor_b(t)
        for p in self.panels:
            p.cursor_b.setVisible(on)
        self._update_b_label()
        self.cursorBMoved.emit(self.cursor_b_t)

    def set_cursor_b(self, t, source=None):
        if self._sync_b:
            return
        self._sync_b = True
        try:
            self.cursor_b_t = float(t)
            for p in self.panels:
                if p is not source:
                    p.cursor_b.setValue(self.cursor_b_t)
        finally:
            self._sync_b = False
        self._update_b_label()
        self.cursorBMoved.emit(self.cursor_b_t)

    def delta_t(self):
        return self.cursor_b_t - self.cursor_t

    def _update_b_label(self):
        dt = self.delta_t()
        text = f"Δt = {dt * 1000:.2f} ms" if abs(dt) < 10 else f"Δt = {dt:.4f} s"
        if dt != 0 and abs(dt) < 1e3:
            text += f"  (1/Δt = {1 / abs(dt):.3g} Hz)"
        for i, p in enumerate(self.panels):
            p.set_b_label(self.b_enabled and i == 0, text)

    def toggle_select(self, panel, curve):
        if curve.selected:
            self.selected = [(p, c) for p, c in self.selected if c is not curve]
            curve.selected = False
        else:
            self.selected.append((panel, curve))
            curve.selected = True
        panel.restyle(curve)
        self.selectionChanged.emit()

    def set_selection(self, pairs):
        for p, c in self.selected:
            c.selected = False
            p.restyle(c)
        self.selected = list(pairs)
        for p, c in self.selected:
            c.selected = True
            p.restyle(c)
        self.selectionChanged.emit()

    def clear_selection(self):
        self.set_selection([])

    def forget(self, panel, curve):
        if any(c is curve for _, c in self.selected):
            self.selected = [(p, c) for p, c in self.selected if c is not curve]
            self.selectionChanged.emit()

    def x_range(self):
        if not self.panels:
            return None
        return tuple(self.panels[0].vb.viewRange()[0])

    def set_x_range(self, a, b):
        self._xsyncing = True
        try:
            for p in self.panels:
                p.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
                p.vb.setXRange(a, b, padding=0)
        finally:
            self._xsyncing = False

    def ensure_visible(self, t):
        r = self.x_range()
        if r and not (r[0] <= t <= r[1]):
            w = r[1] - r[0]
            if t > r[1]:
                self.set_x_range(t - 0.8 * w, t + 0.2 * w)
            else:
                self.set_x_range(t - 0.2 * w, t + 0.8 * w)

    def push_zoom(self):
        r = self.x_range()
        if r is None:
            return
        ys = [(tuple(p.vb.viewRange()[1]), bool(p.vb.state["autoRange"][1])) for p in self.panels]
        self.zoom_stack.append((r, ys))
        del self.zoom_stack[:-50]

    def zoom_back(self):
        if not self.zoom_stack:
            return
        r, ys = self.zoom_stack.pop()
        self.set_x_range(*r)
        for p, (yr, auto) in zip(self.panels, ys):
            if auto:
                p.fit_y()
            else:
                p.vb.setYRange(*yr, padding=0)

    def fit_all(self, push=True):
        if push:
            self.push_zoom()
        lo, hi = np.inf, -np.inf
        for p in self.panels:
            for c in p.curves:
                if len(c.t):
                    lo, hi = min(lo, c.t[0]), max(hi, c.t[-1])
        if math.isfinite(lo) and hi > lo:
            self.set_x_range(lo, hi)
            for p in self.panels:
                p.fit_y()

    def set_events(self, events):
        self.events = events
        self._apply_events()

    def apply_theme(self):
        for p in self.panels:
            p.apply_theme()
        self._apply_events()

    def _apply_events(self):
        for p in self.panels:
            self._apply_events_to(p, self._edge(p, False))

    def _apply_events_to(self, p, labels):
        p.set_events(self.events, labels=labels and self.event_labels)

    def all_curves(self):
        return [(p, c) for p in self.panels for c in p.curves]
