from __future__ import annotations

from pathlib import Path

import re
from collections import OrderedDict

from PySide6.QtCore import QSize, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QLineEdit, QMenu, QStyle, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import math_ops
from ..model import CLOCK_LABELS, EVENTS, VIDEO
from .common import keys_mime
from . import theme
from .file_icons import TAG_H, TAG_W, icon_for_stream

ROLE = Qt.UserRole
GROUP_RE = re.compile(r"^(.*?)(?:_|\[)?(\d+)\]?$")


def group_columns(cols: list[str]) -> "OrderedDict[str, list[str]]":
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for c in cols:
        m = GROUP_RE.match(c)
        g = m.group(1) if m and m.group(1) else c
        groups.setdefault(g, []).append(c)
    out: OrderedDict[str, list[str]] = OrderedDict()
    for g, members in groups.items():
        if len(members) >= 2:
            out[g] = members
        else:
            for c in members:
                out[c] = [c]
    return out


class KeyTree(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setUniformRowHeights(True)

    def item_keys(self, item) -> list[str]:
        d = item.data(0, ROLE) or {}
        return list(d.get("keys", []))

    def mimeData(self, items):
        keys = []
        for it in items:
            for k in self.item_keys(it):
                if k not in keys:
                    keys.append(k)
        return keys_mime(keys)

    def mimeTypes(self):
        from .common import MIME
        return [MIME]


class SignalBrowser(QWidget):
    activated = Signal(str, list)
    folderAction = Signal(str, str, str)
    rootAction = Signal(str, str)

    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search signals (e.g. 005 arm_left tau_ext)")
        self.search.setClearButtonEnabled(True)
        self.tree = KeyTree()
        self.results = KeyTree()
        for t in (self.tree, self.results):
            t.setIconSize(QSize(TAG_W, TAG_H))
        self.results.hide()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.search)
        lay.addWidget(self.tree)
        lay.addWidget(self.results)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._run_search)
        self.search.textChanged.connect(lambda _: self._timer.start())
        self.tree.itemExpanded.connect(self._expand)
        for t in (self.tree, self.results):
            t.itemDoubleClicked.connect(self._double)
            t.setContextMenuPolicy(Qt.CustomContextMenu)
            t.customContextMenuRequested.connect(lambda pos, tr=t: self._menu(tr, pos))
        st = self.style()
        self.icons = {
            "folder": theme.icon("folder", theme.t("warn")),
            "signals": st.standardIcon(QStyle.SP_FileIcon),
            "video": st.standardIcon(QStyle.SP_MediaPlay),
            "events": st.standardIcon(QStyle.SP_FileDialogInfoView),
            "error": st.standardIcon(QStyle.SP_MessageBoxWarning),
            "math": st.standardIcon(QStyle.SP_FileDialogContentsView),
        }

    def rebuild(self):
        expanded = set()
        it = self.tree.invisibleRootItem()

        def walk(item, path):
            for i in range(item.childCount()):
                c = item.child(i)
                p = path + "/" + c.text(0)
                if c.isExpanded():
                    expanded.add(p)
                walk(c, p)
        walk(it, "")
        self.tree.clear()
        dr = self.catalog.derived_root
        if dr.streams:
            ditem = QTreeWidgetItem(["Derived"])
            ditem.setIcon(0, self.icons["math"])
            ditem.setData(0, ROLE, {"type": "derived_root", "keys": []})
            ditem.setToolTip(0, "signals computed with the Math functions")
            self.tree.addTopLevelItem(ditem)
            for s in dr.streams:
                si = QTreeWidgetItem([s.name])
                si.setData(0, ROLE, {"type": "stream", "stream": s.key, "derived": s.rel,
                                     "keys": [s.signal_key(c) for c in s.columns]})
                si.setToolTip(0, s.info.get("note", ""))
                si.setIcon(0, self.icons["math"])
                for c in s.columns:
                    ci = QTreeWidgetItem([c])
                    ci.setData(0, ROLE, {"type": "signal", "keys": [s.signal_key(c)]})
                    si.addChild(ci)
                ditem.addChild(si)
            ditem.setExpanded(True)
        for root in self.catalog.roots:
            ritem = QTreeWidgetItem([root.name])
            ritem.setIcon(0, self.icons["folder"])
            ritem.setData(0, ROLE, {"type": "root", "root": root.name, "keys": []})
            ritem.setToolTip(0, str(root.path))
            self.tree.addTopLevelItem(ritem)
            folders = {"": ritem}
            for s in root.streams:
                parent = self._folder_item(folders, root.name, s.folder)
                if s.info.get("file_group"):
                    parent = self._file_item(folders, root.name, s)
                si = QTreeWidgetItem([s.name])
                si.setData(0, ROLE, {"type": "stream", "stream": s.key,
                                     "keys": [s.key] if s.kind in (EVENTS, VIDEO) else []})
                si.setIcon(0, self.icons["error"] if s.error else icon_for_stream(s))
                tip = [str(s.path), f"kind: {s.kind}", f"clock: {CLOCK_LABELS.get(s.clock, s.clock)}"]
                if s.info.get("rows_estimate"):
                    tip.append(f"~{s.info['rows_estimate']:,} rows")
                if s.info.get("note"):
                    tip.append(s.info["note"])
                if s.error:
                    tip.append(f"error: {s.error}")
                if s.video:
                    tip.append(f"video: {s.video['path'].name}")
                si.setToolTip(0, "\n".join(tip))
                if s.empty or s.error:
                    si.setForeground(0, QBrush(QColor("#999999")))
                    if s.empty:
                        si.setText(0, f"{s.name}  (empty)")
                if s.columns:
                    si.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                parent.addChild(si)
            if root.skipped or root.errors:
                info = QTreeWidgetItem([f"not loaded ({len(root.skipped) + len(root.errors)})"])
                info.setIcon(0, self.icons["error"])
                info.setForeground(0, QBrush(QColor("#999999")))
                for x in root.errors + [f"{p}: no loader" for p in root.skipped]:
                    c = QTreeWidgetItem([x])
                    c.setToolTip(0, x)
                    info.addChild(c)
                ritem.addChild(info)
        def restore(item, path):
            for i in range(item.childCount()):
                c = item.child(i)
                p = path + "/" + c.text(0)
                if p in expanded:
                    c.setExpanded(True)
                restore(c, p)
        restore(it, "")
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if not any(p.startswith("/" + top.text(0)) for p in expanded):
                top.setExpanded(True)

    def _file_item(self, folders, root_name, s):
        """One item per multi-group file (e.g. an hdf5), its groups listed underneath."""
        key = f"{s.folder}/{Path(s.rel).name}" if s.folder else Path(s.rel).name
        if key in folders:
            return folders[key]
        parent = self._folder_item(folders, root_name, s.folder)
        it = QTreeWidgetItem([Path(s.rel).name])
        it.setIcon(0, icon_for_stream(s))
        it.setData(0, ROLE, {"type": "folder", "root": root_name, "folder": s.folder, "keys": []})
        it.setToolTip(0, str(s.path))
        parent.addChild(it)
        folders[key] = it
        return it

    def _folder_item(self, folders, root_name, folder):
        if folder in folders:
            return folders[folder]
        parent_path, _, name = folder.rpartition("/")
        parent = self._folder_item(folders, root_name, parent_path)
        it = QTreeWidgetItem([name])
        it.setIcon(0, self.icons["folder"])
        it.setData(0, ROLE, {"type": "folder", "root": root_name, "folder": folder, "keys": []})
        parent.addChild(it)
        folders[folder] = it
        return it

    def _expand(self, item):
        d = item.data(0, ROLE) or {}
        if d.get("type") != "stream" or item.childCount() or d.get("filled"):
            return
        s = self.catalog.stream(d["stream"])
        if s is None:
            return
        d["filled"] = True
        item.setData(0, ROLE, d)
        nice = s.info.get("col_labels") or {}

        def leaf(m):
            c = QTreeWidgetItem([nice.get(m, m)])
            c.setToolTip(0, m)
            c.setData(0, ROLE, {"type": "signal", "keys": [s.signal_key(m)]})
            if nice.get(m, "").endswith("(empty)"):
                c.setForeground(0, QBrush(QColor("#999999")))
            return c
        for g, members in group_columns(s.columns).items():
            if len(members) == 1:
                item.addChild(leaf(members[0]))
            else:
                gi = QTreeWidgetItem([f"{g}  [{len(members)}]"])
                gi.setData(0, ROLE, {"type": "group", "keys": [s.signal_key(m) for m in members]})
                for m in members:
                    gi.addChild(leaf(m))
                item.addChild(gi)
        sig_keys = [s.signal_key(c) for c in s.columns]
        if s.kind not in (EVENTS, VIDEO) and len(sig_keys) <= 12:
            d["keys"] = sig_keys
            item.setData(0, ROLE, d)

    def _run_search(self):
        q = self.search.text().strip()
        if not q:
            self.results.hide()
            self.tree.show()
            return
        self.tree.hide()
        self.results.show()
        self.results.clear()
        keys = self.catalog.find(q, limit=1500)
        for k in keys:
            s = self.catalog.stream(k)
            col = k.split("::", 1)[1] if "::" in k else ""
            nice = (s.info.get("col_labels") or {}).get(col) if col else None
            text = (f"{s.label} :: {nice}" if nice else f"{s.label} :: {col}") if col else f"{s.label}  ({s.kind})"
            it = QTreeWidgetItem([text])
            it.setToolTip(0, k)
            it.setData(0, ROLE, {"type": "signal" if col else "stream", "stream": s.key, "keys": [k]})
            if not col:
                it.setIcon(0, icon_for_stream(s))
            self.results.addTopLevelItem(it)
        if not keys:
            self.results.addTopLevelItem(QTreeWidgetItem(["no matches"]))

    def _double(self, item, _col):
        d = item.data(0, ROLE) or {}
        t = d.get("type")
        if t in ("signal", "group"):
            self.activated.emit("add", d["keys"])
        elif t == "stream":
            s = self.catalog.stream(d["stream"])
            if s is not None and s.kind == EVENTS:
                self.activated.emit("events", [s.key])
            elif s is not None and s.kind == VIDEO:
                self.activated.emit("video", [s.key])

    def _menu(self, tree, pos):
        item = tree.itemAt(pos)
        if item is None:
            return
        d = item.data(0, ROLE) or {}
        t = d.get("type")
        m = QMenu(self)
        if t in ("signal", "group"):
            keys = []
            for it in tree.selectedItems() or [item]:
                keys += (it.data(0, ROLE) or {}).get("keys", [])
            m.addAction("Add to current panel", lambda: self.activated.emit("add", keys))
            m.addAction("Add to new panel", lambda: self.activated.emit("add_new", keys))
            m.addAction("One panel per signal", lambda: self.activated.emit("add_each", keys))
            add_math_menu(m, keys, lambda op, ks: self.activated.emit(f"math:{op}", ks))
        elif t == "stream":
            s = self.catalog.stream(d["stream"])
            if s is not None:
                if s.kind == EVENTS:
                    m.addAction("Show / hide event markers", lambda: self.activated.emit("events", [s.key]))
                if s.kind == VIDEO:
                    m.addAction("Open video", lambda: self.activated.emit("video", [s.key]))
                if d.get("derived"):
                    m.addAction("Add all to new panel", lambda: self.activated.emit("add_new", d["keys"]))
                    m.addAction("Delete derived signal", lambda: self.activated.emit("del_derived", [d["derived"]]))
                else:
                    m.addAction("Copy path", lambda: self.activated.emit("copy", [str(s.path)]))
                    m.addAction("Time shift this folder…",
                                lambda: self.folderAction.emit("shift", s.root.name, s.folder))
        elif t == "folder":
            m.addAction("Time shift this folder…", lambda: self.folderAction.emit("shift", d["root"], d["folder"]))
            m.addAction("Align folder start to t = 0", lambda: self.folderAction.emit("align", d["root"], d["folder"]))
            m.addAction("Clear time shift", lambda: self.folderAction.emit("clear", d["root"], d["folder"]))
        elif t == "root":
            m.addAction("Rescan", lambda: self.rootAction.emit("rescan", d["root"]))
            m.addAction("Time shift whole folder…", lambda: self.folderAction.emit("shift", d["root"], ""))
            m.addAction("Close folder", lambda: self.rootAction.emit("close", d["root"]))
        if not m.isEmpty():
            m.exec(tree.viewport().mapToGlobal(pos))


def add_math_menu(menu, keys, emit, title="Math"):
    ops = math_ops.available_ops(len(keys))
    if not ops or not keys:
        return None
    sub = menu.addMenu(f"{title} ({len(keys)} signal{'s' if len(keys) != 1 else ''})")
    groups = {}
    for o in ops:
        groups.setdefault(o.group, []).append(o)
    first = True
    for g in ("pose", "basic", "each"):
        if g not in groups:
            continue
        if not first:
            sub.addSeparator()
        first = False
        for o in groups[g]:
            label = o.label + ("  (each)" if o.each and len(keys) > 1 else "")
            sub.addAction(label, lambda o=o: emit(o.key, list(keys)))
    if len(keys) == 2:
        sub.addSeparator()
        sub.addAction("Estimate delay b vs a (cross-correlation)", lambda: emit("delay", list(keys)))
    return sub
