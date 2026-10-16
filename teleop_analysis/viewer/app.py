from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PySide6.QtCore import QElapsedTimer, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressDialog, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget, QWidgetAction)

from .. import math_ops
from ..catalog import Catalog
from ..export import export_view
from ..model import EVENTS, VIDEO, split_signal_key
from . import theme
from .common import PALETTE, STYLES, keys_mime, fmt_value, short_label
from .dialogs import ClockDialog, ExportDialog
from .plots import PlotArea
from .intent_overlay import IntentOverlay, find_gaze_sources, find_local_run, make_gaze_overlay
from .layouts import LayoutPicker
from .tree import SignalBrowser, add_math_menu
from .video_dock import VideoDock

class ValuesTable(QTableWidget):
    def __init__(self, rows, cols, keys_for_rows):
        super().__init__(rows, cols)
        self._keys_for_rows = keys_for_rows
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)

    def mimeData(self, items):
        rows = sorted({it.row() for it in items})
        return keys_mime(self._keys_for_rows(rows), move=True)

    def mimeTypes(self):
        from .common import MIME, MOVE_MIME
        return [MIME, MOVE_MIME]


SPEEDS = ["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x"]
STEP_MODES = ["video frame", "signal sample", "1 ms", "10 ms", "100 ms", "1 s"]


def hint(obj, title, text=None, key=None):
    muted = theme.t("muted")
    k = f"&nbsp;&nbsp;<span style='color:{muted}'>{key}</span>" if key else ""
    body = f"<br>{text}" if text else ""
    obj.setToolTip(f"<b>{title}</b>{k}{body}")
    obj.setStatusTip(f"{title}{f' ({key})' if key else ''}{f' — {text}' if text else ''}")
    return obj


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("teleop-analysis")
        self.resize(1500, 900)
        self.settings = QSettings("teleop-analysis", "viewer")
        self.catalog = Catalog()
        self.catalog.store.progress = self._progress
        self._pd = None
        self.event_streams: list[str] = []
        self.event_rows: list = []
        self.videos: dict[str, VideoDock] = {}
        self.setAcceptDrops(True)
        self.setDockNestingEnabled(True)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self._close_page)
        self.tabs.currentChanged.connect(self._page_changed)
        self.tabs.tabBar().tabBarDoubleClicked.connect(self._rename_page)
        self.tabs.tabBar().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabs.tabBar().customContextMenuRequested.connect(self._tab_menu)
        hint(self.tabs.tabBar(), "Pages", "Double-click to rename, drag to reorder, right-click for more")
        plus = QToolButton()
        theme.set_icon(plus, "plus")
        hint(plus, "New page", None, "Ctrl+T")
        plus.setAutoRaise(True)
        plus.clicked.connect(lambda: self.new_page())
        self.tabs.setCornerWidget(plus, Qt.TopRightCorner)
        self._prev_area = None
        self._page_counter = 0
        self.setCentralWidget(self.tabs)

        self.browser = SignalBrowser(self.catalog)
        self.browser.activated.connect(self._browser_action)
        self.browser.folderAction.connect(self._folder_action)
        self.browser.rootAction.connect(self._root_action)
        self.browser_dock = self._dock("Signals", self.browser, Qt.LeftDockWidgetArea)
        hint(self.browser.search, "Search signals", "All words must match, e.g. 005 arm_left tau_ext")
        self.browser.tree.setStatusTip("Drag signals onto a plot · double-click to add · right-click for more")

        self.values = ValuesTable(0, 5, self._keys_for_rows)
        hint(self.values, "Values at the cursors",
             "Double-click a name to rename it. Drag rows onto another panel to move those signals there")
        self.values.setHorizontalHeaderLabels(["", "signal", "@ A", "@ B", "Δ (B − A)"])
        hh = self.values.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (2, 3, 4):
            hh.setSectionResizeMode(i, QHeaderView.Interactive)
            self.values.setColumnWidth(i, 78)
        self.values.verticalHeader().hide()
        self.values.setWordWrap(False)
        self.values.setTextElideMode(Qt.ElideMiddle)
        self.values.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.values.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.values.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.values.itemChanged.connect(self._value_renamed)
        self.values.cellDoubleClicked.connect(self._value_clicked)
        self.values.itemSelectionChanged.connect(self._table_to_selection)
        self.values.setContextMenuPolicy(Qt.CustomContextMenu)
        self.values.customContextMenuRequested.connect(self._values_menu)
        self.values.setColumnHidden(3, True)
        self.values.setColumnHidden(4, True)
        self._table_sync = False
        self.values_dock = self._dock("Values", self.values, Qt.RightDockWidgetArea)

        ev = QWidget()
        lay = QVBoxLayout(ev)
        lay.setContentsMargins(2, 2, 2, 2)
        self.event_filter = QLineEdit()
        self.event_filter.setPlaceholderText("Filter events")
        hint(self.event_filter, "Filter events", "Show only events containing this text")
        self.event_filter.textChanged.connect(self._fill_events)
        self.event_table = QTableWidget(0, 3)
        self.event_table.setHorizontalHeaderLabels(["t [s]", "source", "event"])
        self.event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.event_table.verticalHeader().hide()
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.event_table.cellClicked.connect(self._event_clicked)
        hint(self.event_table, "Events", "Click an event to move cursor A there")
        lay.addWidget(self.event_filter)
        lay.addWidget(self.event_table)
        self.events_dock = self._dock("Events", ev, Qt.BottomDockWidgetArea)
        self.events_dock.hide()

        self.status = QLabel("")
        self.new_page()
        tip = QLabel("Hover over anything for help  ·  F1 for all controls")
        tip.setObjectName("muted")
        self.statusBar().addWidget(tip)
        self.statusBar().addPermanentWidget(self.status)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._tick)
        self.play_clock = QElapsedTimer()
        self._build_actions()
        self.resizeDocks([self.browser_dock, self.values_dock], [300, 430], Qt.Horizontal)
        self._restore()

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    def _build_actions(self):
        tb = self.addToolBar("main")
        tb.setObjectName("main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolbar = tb
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        vm = mb.addMenu("&View")
        for m in (fm, vm):
            m.setToolTipsVisible(True)

        def act(text, slot, key=None, menu=None, bar=False, tip=None, icon=None, label=None):
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            hint(a, text, tip, key)
            if icon:
                theme.set_icon(a, icon)
            a.triggered.connect(slot)
            if menu is not None:
                menu.addAction(a)
            if bar:
                tb.addAction(a)
                w = tb.widgetForAction(a)
                if label == "":
                    w.setToolButtonStyle(Qt.ToolButtonIconOnly)
                elif label:
                    a.setIconText(label)
            self.addAction(a)
            return a

        act("Open folder…", self.open_folder_dialog, "Ctrl+O", fm, True,
            "Open a log folder; every file it contains appears in the Signals tree", "folder-open-outline",
            "Open")
        self.recent_menu = fm.addMenu("Open recent")
        theme.set_icon(self.recent_menu.menuAction(), "history")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        act("Rescan folders", self.rescan, "F5", fm, True, "Pick up new or changed log files", "refresh", "")
        fm.addSeparator()
        act("Save view…", self.save_view_dialog, "Ctrl+S", fm, False,
            "Save pages, panels, signals and zoom to a .json file", "content-save-outline")
        act("Load view…", self.load_view_dialog, "Ctrl+L", fm, False,
            "Restore a saved view onto the open folders", "folder-upload-outline")
        act("Export view…", self.export_dialog, "Ctrl+E", fm, True,
            "Export the visible window as figure (PNG/PDF) and data (CSV) — exactly what you see",
            "export-variant", "Export")
        fm.addSeparator()
        act("Collect episodes…", self.collect_dialog, None, fm, False,
            "Copy an episode from the avatar, orchestrator and interface logs into a dataset folder",
            "database-export-outline")
        act("Default paths…", self.paths_dialog, None, fm, False,
            "Log roots and dataset folder used as defaults on this machine", "folder-cog-outline")
        fm.addSeparator()
        act("Quit", self.close, "Ctrl+Q", fm, False, None, "exit-to-app")
        tb.addSeparator()
        act("New page", lambda: self.new_page(), "Ctrl+T", vm, False, "New empty page (tab)", "tab-plus")
        self.panel_btn = QToolButton()
        self.panel_btn.setText("Panel")
        self.panel_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        theme.set_icon(self.panel_btn, "view-split-horizontal")
        hint(self.panel_btn, "New panel", "Click: add a panel below. Arrow: pick a panel arrangement", "Ctrl+N")
        self.panel_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.panel_btn.clicked.connect(lambda: self.area.add_panel())
        pmenu = QMenu(self.panel_btn)
        self.picker = LayoutPicker()
        self.picker.picked.connect(lambda spec, new: (pmenu.close(), self.apply_layout(spec, new)))
        wa = QWidgetAction(pmenu)
        wa.setDefaultWidget(self.picker)
        pmenu.addAction(wa)
        self.panel_btn.setMenu(pmenu)
        tb.addWidget(self.panel_btn)
        act("New panel", lambda: self.area.add_panel(), "Ctrl+N", vm, False, "Add an empty plot panel below",
            "view-split-horizontal")
        act("Clear plots", lambda: self.area.clear(), None, vm, True, "Remove all panels of this page",
            "broom", "")
        tb.addSeparator()
        act("Fit all", lambda: self.area.fit_all(), "Home", vm, True,
            "Zoom to all plotted data (or double-click a plot)", "fit-to-screen-outline")
        act("Zoom back", lambda: self.area.zoom_back(), "Backspace", vm, True,
            "Back to the previous zoom", "magnify-minus-outline", "")
        act("t = 0 at cursor", self.zero_at_cursor, None, vm, True,
            "Shift the time axis so that cursor A is t = 0", "timer-outline", "")
        act("Clocks…", self.clock_dialog, None, vm, True, "Clock offsets and per-folder time shifts",
            "clock-outline", "")
        tb.addSeparator()
        act("Step back", lambda: self.step(-1), "Left", vm, True, "Step back by the step size on the right",
            "skip-previous", "")
        self.play_act = act("Play", self.toggle_play, "Space", vm, True, "Play / pause from the cursor", None, "")
        self._play_icon = "play"
        theme.set_icon(self.play_act, "play", "accent_text")
        tb.widgetForAction(self.play_act).setObjectName("playButton")
        act("Step forward", lambda: self.step(1), "Right", vm, True, "Step forward by the step size on the right",
            "skip-next", "")
        self.step_mode = QComboBox()
        self.step_mode.addItems(STEP_MODES)
        hint(self.step_mode, "Step size", "Step size for the step buttons and the Left / Right keys")
        tb.addWidget(self.step_mode)
        self.speed = QComboBox()
        self.speed.addItems(SPEEDS)
        self.speed.setCurrentText("1x")
        hint(self.speed, "Playback speed")
        tb.addWidget(self.speed)
        tb.addSeparator()
        self.b_act = QAction("Δ cursor", self, checkable=True)
        self.b_act.setShortcut(QKeySequence("D"))
        hint(self.b_act, "Δ cursor", "Second time cursor B for measuring Δt (or Shift+click in a plot)", "D")
        theme.set_icon(self.b_act, "arrow-left-right")
        self.b_act.toggled.connect(lambda on: self.area.enable_b(on))
        tb.addAction(self.b_act)
        vm.addAction(self.b_act)
        self.addAction(self.b_act)
        act("Cursors here", self.cursors_to_view, "C", vm, True,
            "Bring cursor A (and B) into the visible window without changing the zoom", "crosshairs-gps", "")
        act("Clear selection", lambda: self.area.clear_selection(), "Esc", vm, False, "Deselect all curves")
        vm.addSeparator()
        self.labels_act = QAction("Event labels", self, checkable=True, checked=True)
        hint(self.labels_act, "Event labels", "Show the event names next to the event lines")
        self.labels_act.toggled.connect(self._toggle_labels)
        vm.addAction(self.labels_act)
        self.dark_act = QAction("Dark theme", self, checkable=True, checked=theme.name() == "dark")
        hint(self.dark_act, "Dark theme", "Switch between the dark and the light look")
        self.dark_act.toggled.connect(lambda on: self.set_theme("dark" if on else "light"))
        vm.addAction(self.dark_act)
        vm.addSeparator()
        for d in (self.browser_dock, self.values_dock, self.events_dock):
            vm.addAction(d.toggleViewAction())
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        th = QToolButton()
        theme.set_icon(th, "theme-light-dark")
        hint(th, "Dark / light theme")
        th.clicked.connect(lambda: self.dark_act.toggle())
        tb.addWidget(th)
        hb = QToolButton()
        theme.set_icon(hb, "help-circle-outline")
        hint(hb, "Controls", "Mouse and keyboard controls", "F1")
        hb.clicked.connect(self._help)
        tb.addWidget(hb)
        hm = mb.addMenu("&Help")
        hm.setToolTipsVisible(True)
        act("Controls", self._help, "F1", hm, False, "Mouse and keyboard controls", "keyboard-outline")
        act("About", self._about, None, hm, False, None, "information-outline")

    def set_theme(self, name):
        theme.apply(name)
        self.settings.setValue("theme", name)
        for i in range(self.tabs.count()):
            self.tabs.widget(i).apply_theme()
        if self.dark_act.isChecked() != (name == "dark"):
            self.dark_act.blockSignals(True)
            self.dark_act.setChecked(name == "dark")
            self.dark_act.blockSignals(False)

    def _about(self):
        QMessageBox.about(self, "teleop-analysis", (
            "<h3>teleop-analysis</h3><p>Time-synced viewer for teleop logs: plot signals, video, gaze and intent "
            "on one timeline, then export.</p>"
            "<p><a href='https://github.com/AlexanderWegenerRobotics/teleop-analysis'>"
            "github.com/AlexanderWegenerRobotics/teleop-analysis</a></p>"))

    def _help(self):
        rows = [
            ("Signals", [("Drag a signal onto a plot", "add it there"),
                         ("Drop on the strip at the bottom", "new panel"),
                         ("Double-click a signal", "add to the highlighted panel"),
                         ("Double-click events / a video", "show them")]),
            ("Zoom and pan", [("Wheel", "zoom time"), ("Shift+wheel, or wheel over the y axis", "zoom y"),
                              ("Left drag", "pan"), ("Right drag", "zoom into the box"),
                              ("Double-click / Home", "fit all"), ("Backspace", "zoom back")]),
            ("Cursors", [("Click", "cursor A"), ("Shift+click", "cursor B (Δt)"), ("D", "toggle cursor B"),
                         ("C", "bring the cursors into view"), ("Right-click empty area", "y cursors (Δy), y label")]),
            ("Curves", [("Hover", "value at that point"), ("Right-click", "remove, rename, colour, math"),
                        ("Ctrl+click / select rows in Values", "select several for math"), ("Esc", "clear selection")]),
            ("Playback", [("Space", "play / pause from the cursor"), ("Left / Right", "step"),
                          ("Video slider", "scrub; plots follow")]),
            ("Export", [("Ctrl+E", "figure + data of exactly the visible window")]),
        ]
        m = theme.t("muted")
        html = ["<table cellspacing='0' cellpadding='3'>"]
        for sec, items in rows:
            html.append(f"<tr><td colspan='2' style='padding-top:10px'><b>{sec}</b></td></tr>")
            for k, v in items:
                html.append(f"<tr><td style='padding-right:18px'>{k}</td><td style='color:{m}'>{v}</td></tr>")
        html.append("</table>")
        box = QMessageBox(self)
        box.setWindowTitle("Controls")
        box.setTextFormat(Qt.RichText)
        box.setText("".join(html))
        box.exec()

    def _restore(self):
        g = self.settings.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        s = self.settings.value("state")
        if s is not None:
            self.restoreState(s)

    def closeEvent(self, ev):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        for d in list(self.videos.values()):
            d.reader.close()
        super().closeEvent(ev)

    @contextmanager
    def busy(self, text=""):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if text:
            self.statusBar().showMessage(text)
        QApplication.processEvents()
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            if self._pd is not None:
                self._pd.close()
                self._pd = None
            self.statusBar().clearMessage()

    def _progress(self, stream, frac):
        if self._pd is None:
            self._pd = QProgressDialog("", None, 0, 100, self)
            self._pd.setWindowTitle("Indexing")
            self._pd.setWindowModality(Qt.ApplicationModal)
            self._pd.setMinimumDuration(300)
        self._pd.setLabelText(f"Indexing {stream.label}\n(only needed the first time a file is opened)")
        self._pd.setValue(int(frac * 100))
        QApplication.processEvents()

    def _remux_progress(self, n):
        self.statusBar().showMessage(f"Preparing video for seeking… {n} frames")
        QApplication.processEvents()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [Path(u.toLocalFile()) for u in ev.mimeData().urls() if u.isLocalFile()]
        for p in paths:
            self.add_folder(p if p.is_dir() else p.parent)

    def open_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open log folder", self.settings.value("last_dir", ""))
        if d:
            self.add_folder(Path(d))

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in self._recent():
            self.recent_menu.addAction(p, lambda p=p: self.add_folder(Path(p)))

    def _recent(self):
        r = self.settings.value("recent", [])
        if isinstance(r, str):
            r = [r]
        return [x for x in (r or []) if x]

    def add_folder(self, path: Path):
        path = Path(path)
        if not path.exists():
            QMessageBox.warning(self, "Not found", f"{path} does not exist.")
            return
        with self.busy(f"Scanning {path} …"):
            root = self.catalog.add_root(path)
        self.settings.setValue("last_dir", str(path.parent))
        rec = [str(path)] + [x for x in self._recent() if x != str(path)]
        self.settings.setValue("recent", rec[:12])
        self.browser.rebuild()
        n = len(root.streams)
        self.statusBar().showMessage(f"{root.name}: {n} streams, {len(root.skipped)} files without loader", 6000)

    def paths_dialog(self):
        from ..config import save_paths
        from .dialogs import PathsDialog
        dlg = PathsDialog(self)
        if dlg.exec() == QDialog.Accepted:
            p = save_paths(dlg.values())
            try:
                opts = json.loads(self.settings.value("collect_opts", "{}") or "{}")
            except (TypeError, ValueError):
                opts = {}
            self.settings.setValue("collect_opts", json.dumps({k: v for k, v in opts.items()
                                                               if k not in ("avatar", "local", "interface", "out")}))
            self.statusBar().showMessage(f"default paths saved to {p}", 5000)

    def collect_dialog(self):
        """File → Collect episodes: GUI front end of teleop_analysis.collect."""
        from ..collect import collect_episode
        from .dialogs import CollectDialog
        try:
            saved = json.loads(self.settings.value("collect_opts", "{}") or "{}")
        except (TypeError, ValueError):
            saved = {}
        dlg = CollectDialog(self, saved)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        self.settings.setValue("collect_opts", json.dumps({k: v[k] for k in ("avatar", "local", "interface", "out", "with_video", "attention")}))
        if not v["out"] or not v["session_id"] or not v["episodes"]:
            QMessageBox.warning(self, "Collect", "Dataset folder, session id and at least one episode are needed.")
            return
        src = {k: Path(v[k]) if v[k] else None for k in ("avatar", "local", "interface")}
        lines, warnings = [], []
        prog = QProgressDialog("Collecting…", "Cancel", 0, len(v["episodes"]), self)
        prog.setWindowModality(Qt.WindowModal)
        for i, ep in enumerate(v["episodes"]):
            prog.setLabelText(f"Episode {ep} …")
            prog.setValue(i)
            QApplication.processEvents()
            if prog.wasCanceled():
                break
            try:
                dst, copied, warn = collect_episode(v["session_id"], ep, Path(v["out"]), src["avatar"], src["local"],
                                                    src["interface"], v["with_video"], v["attention"], None, v["overwrite"])
            except Exception as e:
                warnings.append(f"episode {ep}: {type(e).__name__}: {e}")
                continue
            lines.append(f"{dst.name}: {len(copied)} files")
            warnings += [f"episode {ep}: {w}" for w in warn]
        prog.setValue(len(v["episodes"]))
        msg = "\n".join(lines) or "nothing copied"
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(warnings)
        box = QMessageBox(QMessageBox.Information if not warnings else QMessageBox.Warning, "Collect", msg, parent=self)
        open_btn = box.addButton("Open dataset folder", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is open_btn:
            self.add_folder(Path(v["out"]))

    def rescan(self):
        with self.busy("Rescanning…"):
            self.catalog.reload()
        self.browser.rebuild()
        self._refresh_all()

    def _root_action(self, action, name):
        if action == "close":
            self.catalog.remove_root(name)
            for p, c in [pc for a in self.areas() for pc in a.all_curves()]:
                if self.catalog.stream(c.key) is None:
                    p.remove_curve(c)
            self.event_streams = [k for k in self.event_streams if self.catalog.stream(k)]
            for k in [k for k in self.videos if self.catalog.stream(k) is None]:
                self.videos[k].close()
            self._apply_events()
            self.browser.rebuild()
        elif action == "rescan":
            self.rescan()

    def _folder_action(self, action, root, folder):
        k = (root, folder)
        if action == "shift":
            cur = self.catalog.shifts_s.get(k, 0.0)
            v, ok = QInputDialog.getDouble(self, "Time shift", f"Shift for {root}/{folder or '(all)'} [s]:",
                                           cur, -1e9, 1e9, 4)
            if not ok:
                return
            self.catalog.shifts_s[k] = v
        elif action == "clear":
            self.catalog.shifts_s.pop(k, None)
        elif action == "align":
            r = self.catalog.root(root)
            starts = []
            self.catalog.shifts_s.pop(k, None)
            with self.busy():
                for s in r.streams:
                    if s.folder == folder or s.folder.startswith(folder + "/") or folder == "":
                        if s.kind == EVENTS or s.empty:
                            continue
                        v = self.catalog.stream_start_s(s)
                        if v is not None:
                            starts.append(v)
            if not starts:
                return
            self.catalog.shifts_s[k] = -min(starts)
        self.catalog.invalidate_time()
        self._refresh_all()

    def _browser_action(self, action, keys):
        if action == "add":
            self.add_signals(keys)
        elif action == "add_new":
            self.add_signals(keys, self.area.add_panel())
        elif action == "add_each":
            for k in keys:
                self.add_signals([k], self.area.add_panel())
        elif action == "events":
            for k in keys:
                self.toggle_events(k)
        elif action == "video":
            for k in keys:
                self.open_video(k)
        elif action == "copy":
            QApplication.clipboard().setText("\n".join(keys))
        elif action.startswith("math:"):
            self.apply_math(action[5:], keys)
        elif action == "del_derived":
            for name in keys:
                self.catalog.remove_derived(name)
            self.browser.rebuild()
            self._refresh_all()

    def _load_into(self, panel, keys):
        if panel is None:
            sig = [k for k in keys if split_signal_key(k)[1] is not None]
            if sig:
                panel = self.area.add_panel()
        self.add_signals(keys, panel)

    @property
    def area(self) -> PlotArea:
        return self.tabs.currentWidget()

    def areas(self) -> list:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def new_page(self, name=None, spec=None) -> PlotArea:
        a = PlotArea()
        a.loadRequested.connect(self._load_into)
        a.moveRequested.connect(self._move_into)
        a.cursorMoved.connect(self._cursor_moved)
        a.curvesChanged.connect(self._rebuild_values)
        a.cursorBMoved.connect(self._cursor_b_moved)
        a.curveMenuRequested.connect(self._curve_menu)
        a.selectionChanged.connect(self._selection_to_table)
        a.math_menu_hook = self._fill_selection_math
        a.event_labels = getattr(self, "labels_act", None) is None or self.labels_act.isChecked()
        if self.tabs.count():
            a.set_events(self.area.events)
        self._page_counter += 1
        idx = self.tabs.addTab(a, name or f"Page {self._page_counter}")
        self.tabs.setCurrentIndex(idx)
        if spec is not None:
            a.apply_layout(spec, keep=False)
        return a

    def apply_layout(self, spec, new_page=False):
        if new_page:
            self.new_page(spec=spec)
        else:
            self.area.apply_layout(spec)

    def _page_changed(self, _i):
        new = self.area
        old = self._prev_area
        if new is None:
            return
        if old is not None and old is not new and old in self.areas():
            new.set_cursor(old.cursor_t)
            if old.b_enabled:
                new.enable_b(True, old.cursor_b_t)
            elif new.b_enabled:
                new.enable_b(False)
        self._prev_area = new
        if hasattr(self, "values") and hasattr(self, "b_act"):
            self._rebuild_values()
            self._cursor_b_moved(new.cursor_b_t)

    def _close_page(self, i):
        a = self.tabs.widget(i)
        if a.all_curves():
            r = QMessageBox.question(self, "Close page", f"Close \"{self.tabs.tabText(i)}\" and its plots?")
            if r != QMessageBox.Yes:
                return
        if self.tabs.count() == 1:
            a.clear()
            return
        if a is self._prev_area:
            self._prev_area = None
        self.tabs.removeTab(i)
        a.deleteLater()

    def _rename_page(self, i):
        if i < 0:
            return
        text, ok = QInputDialog.getText(self, "Rename page", "Name:", text=self.tabs.tabText(i))
        if ok and text:
            self.tabs.setTabText(i, text)

    def _tab_menu(self, pos):
        i = self.tabs.tabBar().tabAt(pos)
        m = QMenu(self)
        if i >= 0:
            m.addAction("Rename…", lambda: self._rename_page(i))
            m.addAction("Duplicate", lambda: self._duplicate_page(i))
            m.addAction("Close", lambda: self._close_page(i))
            m.addSeparator()
        m.addAction("New page", lambda: self.new_page())
        m.exec(self.tabs.tabBar().mapToGlobal(pos))

    def _duplicate_page(self, i):
        src = self.tabs.widget(i)
        st = self._area_state(src)
        a = self.new_page(f"{self.tabs.tabText(i)} (copy)")
        self._restore_area(a, st)

    def _keys_for_rows(self, rows):
        allc = self.area.all_curves()
        return [allc[r][1].key for r in rows if r < len(allc)]

    def _move_into(self, target, keys):
        sources = []
        for key in keys:
            src = next(((p, c) for p, c in self.area.all_curves() if c.key == key and p is not target), None)
            if src is None:
                continue
            if target is not None and target.has(key):
                src[0].remove_curve(src[1])
                continue
            p, c = src
            if target is None:
                target = self.area.add_panel()
            visible = c.item.isVisible()
            sources.append(p)
            p.remove_curve(c)
            nc = target.add_curve(c.key, c.t, c.y, c.label, c.color, c.width, c.style)
            if nc is not None:
                nc.item.setVisible(visible)
        if target is not None:
            self.area.set_active(target)
        for p in set(sources):
            if p in self.area.panels and not p.curves and len(self.area.panels) > 1:
                self.area.remove_panel(p)

    def add_signals(self, keys, panel=None, labels=None, colors=None, widths=None, styles=None):
        first = not self.area.all_curves()
        errors = []
        with self.busy("Loading…"):
            for i, key in enumerate(keys):
                s = self.catalog.stream(key)
                skey, col = split_signal_key(key)
                if s is None:
                    errors.append(f"{key}: not found (folder not open or file removed)")
                    continue
                if col is None:
                    if s.kind == EVENTS:
                        self.toggle_events(skey, force=True)
                    elif s.kind == VIDEO:
                        self.open_video(skey)
                    continue
                try:
                    t, y = self.catalog.get(key)
                except Exception as e:
                    errors.append(f"{key}: {type(e).__name__}: {e}")
                    continue
                if not len(t):
                    errors.append(f"{key}: no samples")
                    continue
                if panel is None:
                    panel = self.area.target_panel()
                label = (labels or {}).get(key) or short_label(s, col)
                color = (colors or {}).get(key) or (s.info.get("col_colors") or {}).get(col)
                panel.add_curve(key, t, y, label, color, (widths or {}).get(key),
                                (styles or {}).get(key))
        if errors:
            QMessageBox.warning(self, "Some signals were skipped", "\n".join(errors[:30]))
        if first and self.area.all_curves():
            self.area.fit_all()
            r = self.area.x_range()
            if r and not (r[0] <= self.area.cursor_t <= r[1]):
                self.area.set_cursor(r[0] + 0.02 * (r[1] - r[0]))
        return panel

    def toggle_events(self, key, force=None):
        on = key not in self.event_streams if force is None else force
        if on and key not in self.event_streams:
            self.event_streams.append(key)
        elif not on and key in self.event_streams:
            self.event_streams.remove(key)
        self._apply_events()
        if self.event_streams:
            self.events_dock.show()

    def _apply_events(self):
        rows = []
        for k in self.event_streams:
            s = self.catalog.stream(k)
            if s is None:
                continue
            try:
                t, kinds, labels = self.catalog.events(k)
            except Exception as e:
                self.statusBar().showMessage(f"{k}: {e}", 6000)
                continue
            for ti, ki, li in zip(t, kinds, labels):
                rows.append((float(ti), str(ki), str(li), s.label))
        rows.sort(key=lambda r: r[0])
        self.event_rows = rows
        evs = [(r[0], r[1], f"{r[3]}: {r[2]}") for r in rows]
        for a in self.areas():
            a.set_events(evs)
        self._fill_events()

    def _fill_events(self):
        q = self.event_filter.text().lower().split()
        rows = [r for r in self.event_rows if all(x in f"{r[2]} {r[3]}".lower() for x in q)]
        self.event_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.event_table.setItem(i, 0, QTableWidgetItem(f"{r[0]:.3f}"))
            self.event_table.setItem(i, 1, QTableWidgetItem(r[3]))
            self.event_table.setItem(i, 2, QTableWidgetItem(r[2]))

    def _event_clicked(self, row, _col):
        try:
            t = float(self.event_table.item(row, 0).text())
        except (AttributeError, ValueError):
            return
        self._goto(t)

    def _goto(self, t):
        r = self.area.x_range()
        if r and not (r[0] <= t <= r[1]):
            w = r[1] - r[0]
            self.area.set_x_range(t - w / 2, t + w / 2)
        self.area.set_cursor(t)

    def _toggle_labels(self, on):
        self.area.event_labels = on
        self.area._apply_events()

    def open_video(self, key):
        if key in self.videos:
            self.videos[key].show()
            self.videos[key].raise_()
            return
        s = self.catalog.stream(key)
        if s is None or not s.video:
            return
        try:
            with self.busy(f"Opening {s.video['path'].name} …"):
                d = VideoDock(key, s.label, self.catalog, self, self._remux_progress)
        except Exception as e:
            QMessageBox.warning(self, "Video", f"Cannot open {s.video.get('path')}:\n{e}")
            return
        d.closed.connect(lambda k: self.videos.pop(k, None))
        d.timeRequested.connect(self._video_time)
        run = find_local_run(self.catalog, s)
        if run is not None:
            vname = s.video["path"].name
            d.set_overlay_factory(lambda: IntentOverlay(self.catalog, run, vname),
                                  f"draw object names and the intent module's top phase / target from {run.rel}")
            if self.settings.value("intent_overlay", "false") in (True, "true"):
                d.overlay_btn.setChecked(True)
        d.overlay_btn.toggled.connect(lambda on: self.settings.setValue("intent_overlay", bool(on)))
        gz = find_gaze_sources(self.catalog, s)
        if gz:
            vname = s.video["path"].name
            d.set_overlay_factory(lambda: make_gaze_overlay(self.catalog, gz, vname, s.path.parent),
                                  "draw the operator's gaze from " + " or ".join(Path(g.rel).name for _, g in gz),
                                  name="gaze")
            if self.settings.value("gaze_overlay", "false") in (True, "true"):
                d.gaze_btn.setChecked(True)
        d.gaze_btn.toggled.connect(lambda on: self.settings.setValue("gaze_overlay", bool(on)))
        others = list(self.videos.values())
        self.addDockWidget(Qt.RightDockWidgetArea, d)
        if others:
            self.splitDockWidget(others[-1], d, Qt.Vertical)
        self.videos[key] = d
        d.show_time(self.area.cursor_t)

    def _video_time(self, t):
        self.area.ensure_visible(t)
        self.area.set_cursor(t)

    def _status(self):
        a = self.area.cursor_t
        txt = f"A = {a:.4f} s"
        if self.area.b_enabled:
            b = self.area.cursor_b_t
            dt = b - a
            txt += f"    B = {b:.4f} s    Δt = {dt * 1000:.2f} ms"
        self.status.setText(txt)

    def _cursor_moved(self, t):
        self._status()
        for d in list(self.videos.values()):
            try:
                d.show_time(t)
            except Exception as e:
                d.view.show_image(None, str(e))
        self._update_values()

    def _cursor_b_moved(self, t):
        if self.b_act.isChecked() != self.area.b_enabled:
            self.b_act.blockSignals(True)
            self.b_act.setChecked(self.area.b_enabled)
            self.b_act.blockSignals(False)
        self.values.setColumnHidden(3, not self.area.b_enabled)
        self.values.setColumnHidden(4, not self.area.b_enabled)
        self._status()
        self._update_values()

    def _rebuild_values(self):
        self.values.blockSignals(True)
        rows = self.area.all_curves()
        self.values.setRowCount(len(rows))
        for i, (p, c) in enumerate(rows):
            sw = QTableWidgetItem("  ")
            sw.setBackground(QBrush(QColor(c.color)))
            sw.setFlags(Qt.ItemIsEnabled)
            sw.setToolTip("double-click to change colour")
            self.values.setItem(i, 0, sw)
            it = QTableWidgetItem(c.label)
            it.setToolTip(c.key)
            self.values.setItem(i, 1, it)
            for col in (2, 3, 4):
                v = QTableWidgetItem("")
                v.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.values.setItem(i, col, v)
        self.values.blockSignals(False)
        self._selection_to_table()
        self._update_values()

    def _update_values(self, *_):
        self.values.blockSignals(True)
        a, b = self.area.cursor_t, self.area.cursor_b_t
        for i, (p, c) in enumerate(self.area.all_curves()):
            if self.values.item(i, 2) is None:
                continue
            _, ya = c.value_at(a)
            self.values.item(i, 2).setText(fmt_value(ya) if ya is not None else "—")
            if self.area.b_enabled:
                _, yb = c.value_at(b)
                self.values.item(i, 3).setText(fmt_value(yb) if yb is not None else "—")
                d = (yb - ya) if (ya is not None and yb is not None) else None
                self.values.item(i, 4).setText(fmt_value(d) if d is not None else "—")
        self.values.blockSignals(False)

    def _selection_to_table(self):
        if self._table_sync:
            return
        self._table_sync = True
        try:
            sel = {id(c) for _, c in self.area.selected}
            self.values.clearSelection()
            mode = self.values.selectionMode()
            self.values.setSelectionMode(QAbstractItemView.MultiSelection)
            for i, (p, c) in enumerate(self.area.all_curves()):
                if id(c) in sel:
                    self.values.selectRow(i)
            self.values.setSelectionMode(mode)
        finally:
            self._table_sync = False

    def _table_to_selection(self):
        if self._table_sync:
            return
        self._table_sync = True
        try:
            rows = sorted({ix.row() for ix in self.values.selectedIndexes()})
            allc = self.area.all_curves()
            self.area.set_selection([allc[r] for r in rows if r < len(allc)])
        finally:
            self._table_sync = False

    def _values_menu(self, pos):
        row = self.values.rowAt(pos.y())
        allc = self.area.all_curves()
        if row < 0 or row >= len(allc):
            return
        p, c = allc[row]
        self._curve_menu(p, c, self.values.viewport().mapToGlobal(pos))

    def _selected_keys(self):
        return [c.key for _, c in self.area.selected]

    def _fill_selection_math(self, menu):
        keys = self._selected_keys()
        if not keys:
            menu.addAction("Ctrl+click curves (or select rows in Values) first").setEnabled(False)
            return
        self._add_xy_action(menu)
        sub = add_math_menu(menu, keys, self.apply_math, title="Apply to selection")
        if sub is None:
            menu.addAction("no operation for this many signals").setEnabled(False)

    def _curve_menu(self, panel, curve, pos):
        sel = self.area.selected
        in_sel = any(c is curve for _, c in sel)
        keys = [c.key for _, c in sel] if in_sel and len(sel) > 1 else [curve.key]
        m = QMenu(self)
        head = m.addAction(curve.label)
        head.setEnabled(False)
        _, ya = curve.value_at(self.area.cursor_t)
        m.addAction(f"value @ A: {fmt_value(ya) if ya is not None else '—'}").setEnabled(False)
        m.addSeparator()
        m.addAction("Deselect" if curve.selected else "Select  (Ctrl+click)",
                    lambda: self.area.toggle_select(panel, curve))
        m.addAction("Rename…", lambda: self._rename_curve(panel, curve))
        m.addAction("Colour…", lambda: self._recolor_curve(panel, curve))
        wm = m.addMenu("Line width")
        for w in (0.8, 1.3, 2.0, 3.0):
            a = wm.addAction(f"{w:g}", lambda w=w: panel.set_width(curve, w))
            a.setCheckable(True)
            a.setChecked(abs(curve.width - w) < 1e-6)
        sm = m.addMenu("Line style")
        for k, name in STYLES.items():
            a = sm.addAction(name, lambda k=k: panel.set_style(curve, k))
            a.setCheckable(True)
            a.setChecked(curve.style == k)
        m.addAction("Hide" if curve.item.isVisible() else "Show", lambda: curve.item.setVisible(not curve.item.isVisible()))
        move_keys = keys if in_sel else [curve.key]
        mm = m.addMenu("Move to panel" + (f" ({len(move_keys)} signals)" if len(move_keys) > 1 else ""))
        for i, q in enumerate(self.area.panels):
            if q is panel and len(move_keys) == 1:
                continue
            names = ", ".join(c.label.split(": ")[-1] for c in q.curves[:3]) + (" …" if len(q.curves) > 3 else "")
            title = f"Panel {i + 1}" + (f" – {q.y_label}" if q.y_label else "") + (f"  ({names})" if names else "")
            mm.addAction(title, lambda q=q: self._move_into(q, move_keys))
        mm.addSeparator()
        mm.addAction("New panel", lambda: self._move_into(None, move_keys))
        m.addSeparator()
        add_math_menu(m, keys, self.apply_math, title="Math")
        self._add_xy_action(m)
        m.addSeparator()
        m.addAction("Remove", lambda: panel.remove_curve(curve))
        if in_sel and len(sel) > 1:
            m.addAction(f"Remove all selected ({len(sel)})",
                        lambda: [q.remove_curve(c) for q, c in list(self.area.selected)])
        m.exec(pos)

    def _add_xy_action(self, menu):
        """XY plot of the two selected curves: the first selected is x, the second y."""
        sel = self.area.selected
        if len(sel) != 2:
            menu.addAction("XY plot: select exactly two signals (x first, then y)").setEnabled(False)
            return
        (_, cx), (_, cy) = sel
        menu.addAction(f"XY plot: {cy.label.split(': ')[-1]}  vs  {cx.label.split(': ')[-1]}",
                       lambda: self.open_xy(cx, cy))

    def open_xy(self, cx, cy):
        from .xy_dock import XYDock
        d = XYDock(self.area, cx, cy, self)
        self.addDockWidget(Qt.RightDockWidgetArea, d)
        d.setFloating(True)
        d.resize(520, 440)
        d.show()

    def _rename_curve(self, panel, curve):
        text, ok = QInputDialog.getText(self, "Rename", "Label:", text=curve.label)
        if ok and text:
            panel.rename_curve(curve, text)
            self._rebuild_values()

    def _recolor_curve(self, panel, curve):
        col = QColorDialog.getColor(QColor(curve.color), self)
        if col.isValid():
            panel.recolor_curve(curve, col.name())
            self._rebuild_values()

    def _panel_of(self, key):
        for p, c in self.area.all_curves():
            if c.key == key:
                return p
        return None

    def apply_math(self, op, keys):
        if op == "delay":
            self.estimate_delay(keys)
            return
        o = math_ops.OPS[op]
        params = {}
        if o.params:
            params = self._ask_params(o)
            if params is None:
                return
        if op == "se3":
            keys = math_ops.order_se3(keys)
        elif op == "quat2rpy":
            keys = math_ops.order_quat(keys)
        groups = [[k] for k in keys] if o.each else [keys]
        new_keys = []
        errors = []
        with self.busy("Computing…"):
            for g in groups:
                try:
                    new_keys.append((g, self.catalog.add_derived(op, g, params)))
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
        if errors:
            QMessageBox.warning(self, "Math", "\n".join(errors[:10]))
        if not new_keys:
            return
        self.browser.rebuild()
        for g, out in new_keys:
            if op == "se3":
                self._add_axes_set(out[:3], "position [m]")
                self._add_axes_set(out[3:6], "orientation [deg]")
            elif op == "quat2rpy":
                self._add_axes_set(out, "orientation [deg]")
            else:
                self.add_signals(out, self._panel_of(g[0]) or self.area.target_panel())

    def _add_axes_set(self, keys, ylabel):
        panel = next((p for p in self.area.panels if p.y_label == ylabel), None)
        if panel is None:
            panel = self.area.add_panel()
            panel.set_ylabel(ylabel)
            self.add_signals(keys, panel, colors={k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)})
            return
        used = {c.style for c in panel.curves}
        style = next((s for s in STYLES if s not in used), "solid")
        self.add_signals(keys, panel, colors={k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)},
                         styles={k: style for k in keys})

    def _ask_params(self, o):
        dlg = QDialog(self)
        dlg.setWindowTitle(o.label)
        form = QFormLayout(dlg)
        edits = {}
        for name, label, default in o.params:
            e = QLineEdit(str(default))
            form.addRow(label, e)
            edits[name] = (e, default)
        if o.key == "expr":
            form.addRow(QLabel("Inputs are a, b, c, … in selection order, t is time [s].\n"
                               "Examples:  sqrt(a**2 + b**2)   degrees(arctan2(b, a))   a - np.mean(a)"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if not dlg.exec():
            return None
        out = {}
        for name, (e, default) in edits.items():
            txt = e.text().strip()
            if isinstance(default, float):
                try:
                    out[name] = float(txt)
                except ValueError:
                    QMessageBox.warning(self, "Math", f"{name}: not a number")
                    return None
            else:
                out[name] = txt
        return out

    def estimate_delay(self, keys):
        if len(keys) != 2:
            return
        r = self.area.x_range()
        try:
            t1, y1 = self.catalog.get(keys[0])
            t2, y2 = self.catalog.get(keys[1])
            lag, corr = math_ops.estimate_delay(t1, y1, t2, y2, window=r, max_lag=max(0.5, (r[1] - r[0]) / 4) if r else 1.0)
        except Exception as e:
            QMessageBox.warning(self, "Delay", str(e))
            return
        la, lb = split_signal_key(keys[0])[1], split_signal_key(keys[1])[1]
        box = QMessageBox(self)
        box.setWindowTitle("Delay estimate")
        box.setText(f"b = {lb} lags a = {la} by\n\n    {lag * 1000:.2f} ms\n\n"
                    f"(normalised cross-correlation peak {corr:.3f}, computed on the visible window)")
        shift_btn = box.addButton("Add b shifted back by this", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is shift_btn:
            try:
                out = self.catalog.add_derived("shift", [keys[1]], {"seconds": -lag})
            except Exception as e:
                QMessageBox.warning(self, "Delay", str(e))
                return
            self.browser.rebuild()
            self.add_signals(out, self._panel_of(keys[0]) or self.area.target_panel())

    def _value_renamed(self, item):
        if item.column() != 1:
            return
        rows = self.area.all_curves()
        if item.row() < len(rows):
            p, c = rows[item.row()]
            p.rename_curve(c, item.text())

    def _value_clicked(self, row, col):
        if col != 0:
            return
        rows = self.area.all_curves()
        if row >= len(rows):
            return
        p, c = rows[row]
        col = QColorDialog.getColor(QColor(c.color), self)
        if col.isValid():
            p.recolor_curve(c, col.name())
            self._rebuild_values()

    def _refresh_all(self):
        with self.busy("Updating…"):
            for p, c in [pc for a in self.areas() for pc in a.all_curves()]:
                try:
                    t, y = self.catalog.get(c.key)
                    p.refresh_curve(c, t, y)
                except Exception:
                    p.remove_curve(c)
            for d in self.videos.values():
                d.reload_times()
                d.refresh_overlay()
            self._apply_events()
        self._cursor_moved(self.area.cursor_t)

    def zero_at_cursor(self):
        t = self.area.cursor_t
        self.catalog.origin_ns = int((self.catalog.origin_ns or 0) + round(t * 1e9))
        self.catalog.invalidate_time()
        ranges = [(a, a.x_range()) for a in self.areas()]
        self._refresh_all()
        for a, r in ranges:
            if r:
                a.set_x_range(r[0] - t, r[1] - t)
            a.set_cursor(0.0)

    def clock_dialog(self):
        with self.busy("Estimating clock offsets…"):
            d = ClockDialog(self.catalog, self)
        if d.exec():
            d.apply()
            self._refresh_all()

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_act.setText("Play")
            theme.set_icon(self.play_act, "play", "accent_text")
        else:
            self.play_clock.start()
            self.play_timer.start()
            self.play_act.setText("Pause")
            theme.set_icon(self.play_act, "pause", "accent_text")

    def _tick(self):
        dt = self.play_clock.restart() / 1000.0
        sp = float(self.speed.currentText().rstrip("x"))
        t = self.area.cursor_t + dt * sp
        r = self.area.x_range()
        if r and t > r[1]:
            w = r[1] - r[0]
            self.area.set_x_range(t - 0.1 * w, t + 0.9 * w)
        self.area.set_cursor(t)

    def cursors_to_view(self):
        r = self.area.x_range()
        if not r:
            return
        a, b = r
        w = b - a
        if self.area.b_enabled:
            a_first = self.area.cursor_t <= self.area.cursor_b_t
            ta, tb_ = (a + 0.3 * w, a + 0.7 * w) if a_first else (a + 0.7 * w, a + 0.3 * w)
            self.area.set_cursor(ta)
            self.area.set_cursor_b(tb_)
        else:
            self.area.set_cursor(a + 0.5 * w)

    def _step_times(self):
        mode = self.step_mode.currentText()
        if mode == "video frame":
            for d in self.videos.values():
                if len(d.times) > 1:
                    return d.times
            mode = "signal sample"
        if mode == "signal sample":
            if self.area.selected:
                return self.area.selected[0][1].t
            p = self.area.active if self.area.active in self.area.panels else None
            for q in ([p] if p else []) + self.area.panels:
                if q.curves:
                    return q.curves[0].t
            return None
        return None

    def step(self, direction):
        t = self.area.cursor_t
        times = self._step_times()
        if times is not None and len(times) > 1:
            if direction > 0:
                i = int(np.searchsorted(times, t + 1e-9, side="right"))
                new = float(times[min(i, len(times) - 1)])
            else:
                i = int(np.searchsorted(times, t - 1e-9, side="left")) - 1
                new = float(times[max(i, 0)])
        else:
            unit = {"1 ms": 1e-3, "10 ms": 1e-2, "100 ms": 0.1, "1 s": 1.0}.get(self.step_mode.currentText())
            if unit is None:
                r = self.area.x_range()
                unit = (r[1] - r[0]) / 200 if r else 0.01
            new = t + direction * unit
        self.area.ensure_visible(new)
        self.area.set_cursor(new)

    def _area_state(self, area) -> dict:
        r = area.x_range() or (0.0, 1.0)
        panels = []
        for p in area.panels:
            panels.append({
                "signals": [{"key": c.key, "label": c.label, "color": c.color, "visible": c.item.isVisible(),
                             "width": c.width, "style": c.style} for c in p.curves],
                "y_range": list(p.vb.viewRange()[1]),
                "y_auto": bool(p.vb.state["autoRange"][1]),
                "y_label": p.y_label,
            })
        return {"x_range": [float(r[0]), float(r[1])], "panels": panels, "layout": area.layout_spec()}

    def _restore_area(self, area, st: dict):
        pvs = st.get("panels", [])
        layout = st.get("layout")
        if layout is not None and len(pvs) > 0:
            area.apply_layout(layout, keep=False)
        while len(area.panels) < len(pvs):
            area.add_panel()
        for panel, pv in zip(list(area.panels), pvs):
            sigs = pv.get("signals", [])
            self.add_signals([s["key"] for s in sigs], panel, labels={s["key"]: s.get("label") for s in sigs},
                             colors={s["key"]: s.get("color") for s in sigs},
                             widths={s["key"]: s.get("width") for s in sigs},
                             styles={s["key"]: s.get("style") for s in sigs})
            for c, s in zip(list(panel.curves), sigs):
                c.item.setVisible(s.get("visible", True))
            if pv.get("y_label"):
                panel.set_ylabel(pv["y_label"])
            if not pv.get("y_auto", True) and pv.get("y_range"):
                panel.vb.setYRange(*pv["y_range"], padding=0)
        if st.get("x_range"):
            area.set_x_range(*st["x_range"])

    def view_state(self) -> dict:
        cur = self._area_state(self.area)
        r = cur["x_range"]
        panels = cur["panels"]
        pages = []
        for i, a in enumerate(self.areas()):
            stt = self._area_state(a)
            stt["name"] = self.tabs.tabText(i)
            pages.append(stt)
        return {
            "pages": pages,
            "current_page": self.tabs.currentIndex(),
            "version": 1,
            "roots": [{"name": rt.name, "path": str(rt.path)} for rt in self.catalog.roots],
            "origin_ns": self.catalog.origin_ns,
            "offsets_ms": [[a, b, v] for (a, b), v in self.catalog.user_offsets_ms.items()],
            "shifts_s": [[a, b, v] for (a, b), v in self.catalog.shifts_s.items()],
            "x_range": [float(r[0]), float(r[1])],
            "cursor": self.area.cursor_t,
            "panels": panels,
            "events": list(self.event_streams),
            "videos": list(self.videos.keys()),
            "event_list": [[r_[0], r_[1], r_[2]] for r_ in self.event_rows],
            "derived": self.catalog.derived_defs(),
            "cursor_b": self.area.cursor_b_t if self.area.b_enabled else None,
        }

    def save_view_dialog(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save view", self.settings.value("view_dir", ""),
                                           "View (*.json)")
        if not p:
            return
        Path(p).write_text(json.dumps(self.view_state(), indent=2))
        self.settings.setValue("view_dir", str(Path(p).parent))

    def load_view_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load view", self.settings.value("view_dir", ""),
                                           "View (*.json)")
        if p:
            self.load_view(json.loads(Path(p).read_text()))

    def load_view(self, v: dict):
        open_names = {r.name for r in self.catalog.roots}
        for r in v.get("roots", []):
            if r["name"] in open_names:
                continue
            if Path(r["path"]).exists():
                with self.busy():
                    self.catalog.add_root(Path(r["path"]), r["name"])
        self.browser.rebuild()
        if v.get("origin_ns") is not None:
            self.catalog.origin_ns = int(v["origin_ns"])
        self.catalog.user_offsets_ms = {(a, b): c for a, b, c in v.get("offsets_ms", [])}
        self.catalog.shifts_s = {(a, b): c for a, b, c in v.get("shifts_s", [])}
        self.catalog.invalidate_time()
        missing = []
        for d in v.get("derived", []):
            if d["name"] in self.catalog.derived:
                continue
            try:
                self.catalog.add_derived(d["op"], d["inputs"], d.get("params", {}), name=d["name"])
            except Exception as e:
                missing.append(f"{d['name']}: {e}")
        if missing:
            QMessageBox.warning(self, "Derived signals skipped", "\n".join(missing[:20]))
        self.browser.rebuild()
        for k in list(self.videos):
            self.videos[k].close()
        self.event_streams = []
        self._apply_events()
        self._prev_area = None
        while self.tabs.count() > 1:
            w = self.tabs.widget(self.tabs.count() - 1)
            self.tabs.removeTab(self.tabs.count() - 1)
            w.deleteLater()
        self.area.clear()
        pages = v.get("pages") or [{"name": "Page 1", "panels": v.get("panels", []), "x_range": v.get("x_range")}]
        self._page_counter = 0
        for i, st in enumerate(pages):
            if i == 0:
                self._page_counter = 1
                self.tabs.setTabText(0, st.get("name") or "Page 1")
                a = self.area
            else:
                a = self.new_page(st.get("name"))
            self._restore_area(a, st)
        self.tabs.setCurrentIndex(min(max(int(v.get("current_page", 0)), 0), self.tabs.count() - 1))
        self._prev_area = self.area
        for k in v.get("events", []):
            if self.catalog.stream(k):
                self.toggle_events(k, force=True)
        for k in v.get("videos", []):
            if self.catalog.stream(k):
                self.open_video(k)
        self.area.set_cursor(v.get("cursor", 0.0))
        if v.get("cursor_b") is not None:
            self.area.enable_b(True, v["cursor_b"])

    def export_dialog(self):
        if not self.area.all_curves():
            QMessageBox.information(self, "Export", "Nothing plotted yet.")
            return
        try:
            saved = json.loads(self.settings.value("export_opts", "{}") or "{}")
        except ValueError:
            saved = {}
        d = ExportDialog(self, self.settings.value("export_dir", ""), len(self.area.panels), len(self.videos),
                         self.area.b_enabled, saved)
        if not d.exec():
            return
        o = d.options()
        self.settings.setValue("export_opts", json.dumps(d.remembered()))
        if not (o["formats"] or o["data"] or o["view_json"] or o["video"]):
            return
        view = self.view_state()
        for p in view["panels"]:
            p["signals"] = [s for s in p["signals"] if s.get("visible", True)]
        frames = []
        if o["video"]:
            for vd in self.videos.values():
                vd.show_time(self.area.cursor_t)
                if vd.last_img is not None:
                    frames.append((vd.windowTitle(), vd.export_image(), vd.last_t))
        cursors = []
        if o["cursors"] or (o["video"] and o["video_mode"] == 0):
            cursors.append((self.area.cursor_t, "#e41a1c", "A"))
            if self.area.b_enabled and o["cursors"]:
                cursors.append((self.area.cursor_b_t, "#1f5fbf", "B"))
        with self.busy("Exporting…"):
            try:
                out = export_view(self.catalog, view, o["base"], o["formats"], o["width_cm"],
                                  o["panel_height_cm"], o["dpi"], o["events"], o["data"], o["title"],
                                  view_json=o["view_json"], frames=frames,
                                  frames_mode="inside" if o["video_mode"] == 0 else "separate", cursors=cursors)
            except Exception as e:
                QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}")
                return
        self.settings.setValue("export_dir", str(Path(o["base"]).parent))
        QMessageBox.information(self, "Exported", "\n".join(str(p) for p in out))


def _excepthook(win):
    def hook(t, v, tb):
        msg = "".join(traceback.format_exception(t, v, tb))
        sys.stderr.write(msg)
        try:
            QMessageBox.critical(win, "Unexpected error", msg[-3000:])
        except Exception:
            pass
    return hook


def main(argv=None):
    ap = argparse.ArgumentParser(prog="teleop-analysis", description="Time-synchronised log viewer")
    ap.add_argument("folders", nargs="*", help="log folders to open")
    ap.add_argument("--view", help="view .json to load")
    ap.add_argument("--install-shortcut", action="store_true",
                    help="add a start-menu entry (searchable, pinnable to the taskbar) and exit")
    args = ap.parse_args(argv)
    from .. import shortcut
    if args.install_shortcut:
        shortcut.main([])
        return 0
    if sys.stderr is None or sys.stdout is None:
        from ..cache import cache_dir
        log = open(cache_dir().parent / "viewer.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    shortcut.set_app_id()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("teleop-analysis")
    theme.apply(str(QSettings("teleop-analysis", "viewer").value("theme", "dark") or "dark"), app)
    if shortcut.ICON.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(shortcut.ICON)))
    win = MainWindow()
    sys.excepthook = _excepthook(win)
    win.show()
    for f in args.folders:
        win.add_folder(Path(f))
    if args.view:
        win.load_view(json.loads(Path(args.view).read_text()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
