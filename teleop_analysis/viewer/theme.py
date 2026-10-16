"""Dark / light theme: palette tokens, Qt style sheet, pyqtgraph colours and toolbar icons."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

THEMES = {
    "dark": {
        "bg": "#0f141c",
        "surface": "#151b25",
        "surface2": "#1c2431",
        "raised": "#232d3c",
        "border": "#2a3445",
        "text": "#e3e8f0",
        "muted": "#8a96a8",
        "accent": "#38bdf8",
        "accent_text": "#06202c",
        "accent_soft": "#173447",
        "warn": "#ff9f1a",
        "plot_bg": "#0c1118",
        "plot_fg": "#9aa6b8",
        "overlay": (21, 27, 37, 225),
        "cursor_a": "#ff5a5f",
        "cursor_b": "#5ab0ff",
        "ycursor": "#4ade80",
        "palette": ["#38bdf8", "#ff9f1a", "#4ade80", "#ff5a5f", "#c084fc", "#f2c94c", "#f472b6", "#a3b1c6",
                    "#2dd4bf", "#fb923c"],
    },
    "light": {
        "bg": "#eef1f5",
        "surface": "#ffffff",
        "surface2": "#f4f6f9",
        "raised": "#e8ecf2",
        "border": "#d5dbe4",
        "text": "#1b2330",
        "muted": "#677386",
        "accent": "#0284c7",
        "accent_text": "#ffffff",
        "accent_soft": "#dcefff",
        "warn": "#d97706",
        "plot_bg": "#ffffff",
        "plot_fg": "#3b4656",
        "overlay": (255, 255, 255, 230),
        "cursor_a": "#e0262b",
        "cursor_b": "#1f5fbf",
        "ycursor": "#1f9d3a",
        "palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf"],
    },
}

_current = {"name": "dark"}
_listeners = []


def name() -> str:
    return _current["name"]


def t(key: str):
    return THEMES[_current["name"]][key]


def on_change(fn):
    _listeners.append(fn)


def icon(name_: str, color: str | None = None):
    try:
        import qtawesome as qta
        c = color or t("text")
        return qta.icon(f"mdi6.{name_}", color=c, color_disabled=t("muted"), color_active=t("accent"))
    except Exception:
        from PySide6.QtGui import QIcon
        return QIcon()


_icon_targets = []


def set_icon(obj, name_: str, color_key: str = "text"):
    _icon_targets.append((obj, name_, color_key))
    obj.setIcon(icon(name_, t(color_key)))
    return obj


def _refresh_icons():
    keep = []
    for obj, n, ck in _icon_targets:
        try:
            obj.setIcon(icon(n, t(ck)))
            keep.append((obj, n, ck))
        except RuntimeError:
            pass
    _icon_targets[:] = keep


def _svg(name_: str, body: str) -> str:
    import tempfile
    from pathlib import Path
    d = Path(tempfile.gettempdir()) / "teleop-analysis-theme"
    d.mkdir(exist_ok=True)
    p = d / f"{name_}.svg"
    p.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16'>{body}</svg>")
    return p.as_posix()


def _close_svgs(c) -> tuple[str, str]:
    x = "<path d='M5 5 L11 11 M11 5 L5 11' stroke='{}' stroke-width='1.6' stroke-linecap='round'/>"
    normal = _svg(f"close-{c['muted'][1:]}", x.format(c["muted"]))
    hover = _svg(f"close-hover-{c['text'][1:]}", f"<rect x='1' y='1' width='14' height='14' rx='4' "
                 f"fill='{c['raised']}'/>" + x.format(c["text"]))
    return normal, hover


def _qss(c) -> str:
    close, close_hover = _close_svgs(c)
    check = _svg(f"check-{c['accent_text'][1:]}", "<path d='M3.5 8.5 L6.5 11.5 L12.5 4.5' fill='none' "
                 f"stroke='{c['accent_text']}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>")
    return f"""
QCheckBox::indicator, QGroupBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
    border: 1.5px solid {c['muted']}; background: {c['surface2']}; }}
QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; image: url({check}); }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::close-button {{ image: url({close}); subcontrol-position: right; margin-left: 4px; }}
QTabBar::close-button:hover {{ image: url({close_hover}); }}
QWidget {{ color: {c['text']}; }}
QMainWindow, QDialog {{ background: {c['bg']}; }}
QMainWindow::separator {{ background: {c['bg']}; width: 6px; height: 6px; }}
QMainWindow::separator:hover {{ background: {c['accent']}; }}
QToolTip {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border']};
    padding: 6px 8px; border-radius: 6px; }}
QMenuBar {{ background: {c['bg']}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {c['raised']}; }}
QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 6px 26px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QMenu::item:disabled {{ color: {c['muted']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 8px; }}
QMenu::icon {{ padding-left: 6px; }}
QToolBar {{ background: {c['bg']}; border: none; padding: 6px 8px; spacing: 4px; }}
QToolBar::separator {{ background: {c['border']}; width: 1px; margin: 6px 8px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 9px; }}
QToolButton:hover {{ background: {c['raised']}; border-color: {c['border']}; }}
QToolButton:pressed {{ background: {c['accent_soft']}; }}
QToolButton:checked {{ background: {c['accent_soft']}; border-color: {c['accent']}; color: {c['accent']}; }}
QToolButton[popupMode="1"] {{ padding-right: 20px; }}
QToolButton::menu-button {{ border: none; width: 16px; }}
QToolButton#playButton {{ background: {c['accent']}; border-radius: 16px; padding: 6px; }}
QToolButton#playButton:hover {{ background: {c['accent']}; border: 2px solid {c['text']}; }}
QPushButton {{ background: {c['raised']}; border: 1px solid {c['border']}; border-radius: 7px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {c['accent']}; }}
QPushButton:default {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']}; }}
QPushButton#browse {{ padding: 5px 7px; }}
QPushButton:disabled {{ color: {c['muted']}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {c['surface2']}; border: 1px solid {c['border']}; border-radius: 7px; padding: 5px 8px;
    selection-background-color: {c['accent']}; selection-color: {c['accent_text']}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']};
    selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; outline: 0; padding: 4px; }}
QDockWidget {{ titlebar-close-icon: none; font-weight: 600; }}
QDockWidget::title {{ background: {c['bg']}; padding: 8px 10px 6px 12px; text-align: left; }}
QDockWidget > QWidget {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 10px; }}
QTreeView, QTreeWidget, QTableView, QTableWidget, QListView {{ background: {c['surface']}; border: none;
    alternate-background-color: {c['surface2']}; gridline-color: {c['border']}; outline: 0;
    selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}
QTreeView::item, QTableView::item {{ padding: 3px 2px; border-radius: 4px; }}
QTreeView::item:hover, QTableView::item:hover {{ background: {c['surface2']}; }}
QTreeView::item:selected, QTableView::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QHeaderView::section {{ background: {c['surface2']}; color: {c['muted']}; border: none;
    border-bottom: 1px solid {c['border']}; padding: 6px 8px; font-weight: 600; }}
QTableCornerButton::section {{ background: {c['surface2']}; border: none; }}
QTabWidget::pane {{ border: none; background: {c['surface']}; }}
QTabBar {{ background: {c['bg']}; }}
QTabBar::tab {{ background: transparent; color: {c['muted']}; padding: 8px 16px; margin-right: 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: {c['surface']}; color: {c['text']}; border-bottom: 2px solid {c['accent']}; }}
QTabBar::tab:hover:!selected {{ background: {c['surface2']}; color: {c['text']}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle {{ background: {c['raised']}; border-radius: 4px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {c['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['raised']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c['text']}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {c['accent']}; }}
QSplitter::handle {{ background: {c['bg']}; }}
QSplitter::handle:hover {{ background: {c['accent_soft']}; }}
QSplitter::handle:pressed {{ background: {c['accent']}; }}
QStatusBar {{ background: {c['bg']}; color: {c['muted']}; }}
QStatusBar::item {{ border: none; }}
QCheckBox, QRadioButton {{ spacing: 7px; }}
QProgressBar {{ background: {c['surface2']}; border: 1px solid {c['border']}; border-radius: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 5px; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#dropStrip {{ color: {c['muted']}; border: 1.5px dashed {c['border']}; border-radius: 10px;
    background: {c['surface2']}; margin: 4px 6px 6px 6px; }}
QLabel#dropStrip[hot="true"] {{ color: {c['accent']}; border-color: {c['accent']}; background: {c['accent_soft']}; }}
QLabel#videoView {{ background: #05070a; color: {c['muted']}; border-radius: 8px; }}
"""


def _palette(c) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.Window: c["bg"], QPalette.WindowText: c["text"], QPalette.Base: c["surface"],
        QPalette.AlternateBase: c["surface2"], QPalette.Text: c["text"], QPalette.Button: c["raised"],
        QPalette.ButtonText: c["text"], QPalette.Highlight: c["accent"], QPalette.HighlightedText: c["accent_text"],
        QPalette.ToolTipBase: c["raised"], QPalette.ToolTipText: c["text"], QPalette.PlaceholderText: c["muted"],
        QPalette.Link: c["accent"], QPalette.BrightText: c["warn"],
    }
    for role, col in roles.items():
        p.setColor(role, QColor(col))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(c["muted"]))
    return p


def apply(theme: str = "dark", app: QApplication | None = None):
    from .common import PALETTE
    theme = theme if theme in THEMES else "dark"
    _current["name"] = theme
    c = THEMES[theme]
    app = app or QApplication.instance()
    if app is not None:
        app.setStyle("Fusion")
        f = QFont()
        f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial"])
        f.setPointSizeF(10)
        app.setFont(f)
        app.setPalette(_palette(c))
        app.setStyleSheet(_qss(c))
    pg.setConfigOptions(background=c["plot_bg"], foreground=c["plot_fg"])
    PALETTE[:] = c["palette"]
    _refresh_icons()
    for fn in list(_listeners):
        try:
            fn()
        except RuntimeError:
            _listeners.remove(fn)
