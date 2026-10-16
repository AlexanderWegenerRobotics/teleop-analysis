from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QCheckBox, QGridLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from . import theme

P = None

LAYOUTS = [
    ("Single panel", P),
    ("Top / bottom", ("v", [P, P])),
    ("Three stacked", ("v", [P, P, P])),
    ("Four stacked", ("v", [P, P, P, P])),
    ("Left / right", ("h", [P, P])),
    ("Three side by side", ("h", [P, P, P])),
    ("2 × 2 grid", ("v", [("h", [P, P]), ("h", [P, P])])),
    ("3 × 2 grid", ("v", [("h", [P, P, P]), ("h", [P, P, P])])),
    ("Two on top, big one below", ("v", [("h", [P, P]), P], [1, 2])),
    ("Big one on top, two below", ("v", [P, ("h", [P, P])], [2, 1])),
    ("Big one left, two right", ("h", [P, ("v", [P, P])], [2, 1])),
    ("Two left, big one right", ("h", [("v", [P, P]), P], [1, 2])),
]


def _rects(spec, r: QRectF, out: list):
    if not isinstance(spec, (list, tuple)):
        out.append(r)
        return
    orient, children = spec[0], spec[1]
    w = list(spec[2]) if len(spec) > 2 and spec[2] else [1] * len(children)
    tot = float(sum(w)) or 1.0
    pos = 0.0
    for ch, wi in zip(children, w):
        f0, f1 = pos / tot, (pos + wi) / tot
        pos += wi
        if orient == "v":
            sub = QRectF(r.left(), r.top() + f0 * r.height(), r.width(), (f1 - f0) * r.height())
        else:
            sub = QRectF(r.left() + f0 * r.width(), r.top(), (f1 - f0) * r.width(), r.height())
        _rects(ch, sub, out)


def layout_icon(spec, w: int = 56, h: int = 40) -> QIcon:
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    qp = QPainter(pm)
    qp.setRenderHint(QPainter.Antialiasing, True)
    qp.setPen(QPen(QColor(theme.t("border")), 1))
    qp.setBrush(QColor(theme.t("surface2")))
    qp.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 4, 4)
    rects = []
    _rects(spec, QRectF(3, 3, w - 6, h - 6), rects)
    qp.setPen(QPen(QColor(theme.t("accent")), 1))
    qp.setBrush(QColor(theme.t("accent_soft")))
    for r in rects:
        qp.drawRoundedRect(r.adjusted(1, 1, -1, -1), 2, 2)
    qp.end()
    return QIcon(pm)


def count_panels(spec) -> int:
    out = []
    _rects(spec, QRectF(0, 0, 1, 1), out)
    return len(out)


class LayoutPicker(QWidget):
    picked = Signal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        title = QLabel("Arrange panels")
        title.setObjectName("muted")
        lay.addWidget(title)
        grid = QGridLayout()
        grid.setSpacing(4)
        self.buttons = []
        for i, (name, spec) in enumerate(LAYOUTS):
            b = QToolButton()
            b.setIcon(layout_icon(spec))
            b.setIconSize(QSize(56, 40))
            b.setToolTip(name)
            b.setAutoRaise(True)
            b.clicked.connect(lambda _=False, s=spec: self.picked.emit(s, self.new_page.isChecked()))
            grid.addWidget(b, i // 4, i % 4)
            self.buttons.append((b, spec))
        lay.addLayout(grid)
        self.new_page = QCheckBox("open as a new page")
        self.new_page.setToolTip("off: re-arrange the current page (its plots are kept, in order)\n"
                                 "on: create a new empty page with this layout")
        lay.addWidget(self.new_page)
        theme.on_change(self.refresh_icons)

    def refresh_icons(self):
        for b, spec in self.buttons:
            b.setIcon(layout_icon(spec))
