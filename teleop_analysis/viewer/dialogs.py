from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from . import theme
from ..model import CLOCK_INTERFACE, CLOCK_LABELS, CLOCK_RELATIVE


def browse_button():
    b = QPushButton()
    b.setObjectName("browse")
    theme.set_icon(b, "folder-open-outline")
    b.setToolTip("Browse…")
    return b


class ClockDialog(QDialog):
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clocks and time shifts")
        self.catalog = catalog
        self.resize(720, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "All data is shown on the interface / operator-PC clock. Offsets are added to a clock's timestamps.\n"
            "The avatar (NUC) offset is estimated from command.csv (round trip vs. one-way transit). "
            "Edit a value to override it; clear the cell to go back to the estimate."))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["folder", "clock", "estimated [ms]", "used offset [ms]"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.table)
        self.rows = []
        for root in catalog.roots:
            clocks = sorted({s.clock for s in root.streams} - {CLOCK_INTERFACE, CLOCK_RELATIVE})
            for clock in clocks:
                est = catalog.estimate_offset_ms(root, clock)
                r = self.table.rowCount()
                self.table.insertRow(r)
                for i, txt in enumerate([root.name, CLOCK_LABELS.get(clock, clock),
                                         "—" if est is None else f"{est:.2f}"]):
                    it = QTableWidgetItem(txt)
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(r, i, it)
                user = catalog.user_offsets_ms.get((root.name, clock))
                self.table.setItem(r, 3, QTableWidgetItem("" if user is None else f"{user:.3f}"))
                self.rows.append((root.name, clock))
        lay.addWidget(QLabel("Folder time shifts [s] (right-click a folder in the tree to add one):"))
        self.shifts = QTableWidget(0, 3)
        self.shifts.setHorizontalHeaderLabels(["folder", "subfolder", "shift [s]"])
        self.shifts.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for (rn, f), v in sorted(catalog.shifts_s.items()):
            r = self.shifts.rowCount()
            self.shifts.insertRow(r)
            for i, txt in enumerate([rn, f or "(all)"]):
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.shifts.setItem(r, i, it)
            self.shifts.setItem(r, 2, QTableWidgetItem(f"{v:.6f}"))
        lay.addWidget(self.shifts)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def apply(self):
        for r, k in enumerate(self.rows):
            txt = (self.table.item(r, 3).text() if self.table.item(r, 3) else "").strip()
            if not txt:
                self.catalog.user_offsets_ms.pop(k, None)
            else:
                try:
                    self.catalog.user_offsets_ms[k] = float(txt)
                except ValueError:
                    pass
        new = {}
        for r in range(self.shifts.rowCount()):
            rn = self.shifts.item(r, 0).text()
            f = self.shifts.item(r, 1).text()
            f = "" if f == "(all)" else f
            try:
                v = float(self.shifts.item(r, 2).text())
            except ValueError:
                continue
            if v != 0:
                new[(rn, f)] = v
        self.catalog.shifts_s = new
        self.catalog.invalidate_time()


class ExportDialog(QDialog):
    def __init__(self, parent=None, last_dir="", n_panels=1, n_videos=0, has_b=False, saved=None):
        super().__init__(parent)
        self.setWindowTitle("Export current view")
        saved = saved or {}
        form = QFormLayout(self)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit(str(Path(last_dir or Path.home()) / "plot"))
        btn = browse_button()
        btn.clicked.connect(self._browse)
        h.addWidget(self.path, 1)
        h.addWidget(btn)
        form.addRow("File base name", row)
        self.png = QCheckBox("png")
        self.pdf = QCheckBox("pdf")
        self.svg = QCheckBox("svg")
        self.png.setChecked(saved.get("png", True))
        self.pdf.setChecked(saved.get("pdf", False))
        self.svg.setChecked(saved.get("svg", False))
        fr = QWidget()
        fh = QHBoxLayout(fr)
        fh.setContentsMargins(0, 0, 0, 0)
        for w in (self.png, self.pdf, self.svg):
            fh.addWidget(w)
        fh.addStretch(1)
        form.addRow("Figure", fr)
        self.width = QDoubleSpinBox()
        self.width.setRange(4, 60)
        self.width.setValue(saved.get("width_cm", 16))
        self.width.setSuffix(" cm")
        form.addRow("Width (plots)", self.width)
        self.height = QDoubleSpinBox()
        self.height.setRange(1.5, 30)
        self.height.setValue(saved.get("panel_height_cm", 4.5 if n_panels > 1 else 6))
        self.height.setSuffix(" cm")
        form.addRow("Height per panel", self.height)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 600)
        self.dpi.setValue(saved.get("dpi", 200))
        form.addRow("PNG dpi", self.dpi)
        self.title = QLineEdit()
        form.addRow("Title (optional)", self.title)
        self.events = QCheckBox("Event markers")
        self.events.setChecked(saved.get("events", True))
        form.addRow("", self.events)
        self.cursors = QCheckBox("Draw cursor A" + (" and B" if has_b else ""))
        self.cursors.setChecked(saved.get("cursors", False))
        form.addRow("", self.cursors)
        vrow = QWidget()
        vh = QHBoxLayout(vrow)
        vh.setContentsMargins(0, 0, 0, 0)
        self.video = QCheckBox("Video frame at cursor A")
        self.video_mode = QComboBox()
        self.video_mode.addItems(["next to the plots", "separate image"])
        self.video_mode.setCurrentIndex(saved.get("video_mode", 0))
        self.video.setChecked(saved.get("video", False) and n_videos > 0)
        self.video.setEnabled(n_videos > 0)
        self.video_mode.setEnabled(n_videos > 0)
        if n_videos == 0:
            self.video.setToolTip("open a video first")
        vh.addWidget(self.video)
        vh.addWidget(self.video_mode, 1)
        form.addRow("", vrow)
        self.data = QCheckBox("Data as CSV")
        self.data.setChecked(saved.get("data", False))
        self.data.setToolTip("All visible signals of all panels, only the visible time window, at their own\n"
                             "sample times. Writes _data.csv (long: signal, t_s, value) and _wide.csv\n"
                             "(one column per signal, empty where a signal has no sample).")
        form.addRow("", self.data)
        self.view_json = QCheckBox("View file (.json, can be loaded again)")
        self.view_json.setChecked(saved.get("view_json", False))
        form.addRow("", self.view_json)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export base name", self.path.text())
        if p:
            self.path.setText(str(Path(p).with_suffix("")))

    def options(self) -> dict:
        fmts = [f for f, w in (("png", self.png), ("pdf", self.pdf), ("svg", self.svg)) if w.isChecked()]
        return dict(base=self.path.text(), formats=fmts, width_cm=self.width.value(),
                    panel_height_cm=self.height.value(), dpi=self.dpi.value(), events=self.events.isChecked(),
                    data=self.data.isChecked(), title=self.title.text(), view_json=self.view_json.isChecked(),
                    video=self.video.isChecked(), video_mode=self.video_mode.currentIndex(),
                    cursors=self.cursors.isChecked())

    def remembered(self) -> dict:
        o = self.options()
        keep = {k: o[k] for k in ("width_cm", "panel_height_cm", "dpi", "events", "data", "view_json", "video",
                                  "video_mode", "cursors")}
        keep.update(png=self.png.isChecked(), pdf=self.pdf.isChecked(), svg=self.svg.isChecked())
        return keep


class CollectDialog(QDialog):
    """Collect one or more episodes from the avatar, orchestrator and interface logs into a dataset folder.

    Any of the three log fields accepts the log root, a <session_id>/ folder or a <NNN>/ episode folder:
    the root is kept, session and episode are filled in from the rest of the path."""

    FOLDERS = (("avatar", "Avatar logs"), ("local", "Orchestrator logs"), ("interface", "Interface logs"),
               ("out", "Dataset folder"))

    def __init__(self, parent=None, saved=None):
        super().__init__(parent)
        self.setWindowTitle("Collect episodes")
        self.setMinimumWidth(720)
        saved = {**self.defaults(), **{k: v for k, v in (saved or {}).items() if v}}
        form = QFormLayout(self)
        hint = QLabel("Paste the log root, a session folder or an episode folder into any field; "
                      "the session and episode are read from the path. Missing roots are guessed from the others.")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.paths: dict[str, QLineEdit] = {}
        for key, label in self.FOLDERS:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(saved.get(key, ""))
            btn = browse_button()
            btn.clicked.connect(lambda _=False, k=key: self._browse(k))
            h.addWidget(edit, 1)
            h.addWidget(btn)
            form.addRow(label, row)
            self.paths[key] = edit
            if key != "out":
                edit.editingFinished.connect(lambda k=key: self._path_edited(k))
        self.session = QComboBox()
        self.session.setEditable(True)
        self.session.currentTextChanged.connect(self._refresh_episodes)
        form.addRow("Session id", self.session)
        self.episodes = QLineEdit()
        self.episodes.setPlaceholderText("e.g. 3 4 5  (empty = every episode of the session)")
        form.addRow("Episodes", self.episodes)
        self.found = QLabel("")
        self.found.setWordWrap(True)
        form.addRow("", self.found)
        self.with_video = QCheckBox("copy videos (avatar .h264, interface session*.mp4)")
        self.with_video.setChecked(bool(saved.get("with_video", False)))
        self.attention = QCheckBox("render session_attention.mp4 when the interface did not write one")
        self.attention.setChecked(bool(saved.get("attention", False)))
        self.overwrite = QCheckBox("overwrite files already in the dataset")
        for w in (self.with_video, self.attention, self.overwrite):
            form.addRow("", w)
        self.with_video.toggled.connect(self.attention.setEnabled)
        self.attention.setEnabled(self.with_video.isChecked())
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Collect")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        for key in ("avatar", "local", "interface"):
            self._path_edited(key, refresh=False)
        self._refresh_sessions()

    @staticmethod
    def defaults() -> dict:
        from ..config import load_paths
        return {k: v for k, v in load_paths().items() if v}

    def _browse(self, key):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", self.paths[key].text())
        if d:
            self.paths[key].setText(d)
            self._path_edited(key)

    def _path_edited(self, key, refresh=True):
        """Normalises a pasted path to its log root and picks up session / episode / sibling roots from it."""
        from ..collect import resolve_path, sibling_defaults
        text = self.paths[key].text().strip()
        if not text:
            return
        root, session, episode = resolve_path(text)
        self.paths[key].setText(str(root))
        for k, v in sibling_defaults(root).items():
            if not self.paths[k].text().strip():
                self.paths[k].setText(v)
        if refresh:
            self._refresh_sessions()
        if session:
            self.session.setCurrentText(session)
        if episode is not None:
            self.episodes.setText(str(episode))

    def _folder(self, key) -> Path | None:
        t = self.paths[key].text().strip()
        return Path(t) if t else None

    def _refresh_sessions(self):
        ids = set()
        for key in ("local", "interface"):
            f = self._folder(key)
            if f and f.is_dir():
                ids |= {p.name for p in f.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) > 4}
        cur = self.session.currentText()
        self.session.blockSignals(True)
        self.session.clear()
        self.session.addItems(sorted(ids, reverse=True))
        if cur:
            self.session.setCurrentText(cur)
        self.session.blockSignals(False)
        self._refresh_episodes()

    def session_episodes(self) -> list[int]:
        """Episode indices present for the chosen session in the orchestrator or interface logs."""
        sid = self.session.currentText().strip()
        eps = set()
        for key in ("local", "interface"):
            f = self._folder(key)
            if sid and f and (f / sid).is_dir():
                eps |= {int(p.name) for p in (f / sid).iterdir() if p.is_dir() and p.name.isdigit()}
        return sorted(eps)

    def _refresh_episodes(self):
        eps = self.session_episodes()
        self.found.setText(f"episodes of this session: {', '.join(map(str, eps))}" if eps
                           else "no episodes found for this session in the orchestrator / interface logs")

    def episode_list(self) -> list[int]:
        t = self.episodes.text().replace(",", " ").split()
        return [int(x) for x in t] if t else self.session_episodes()

    def values(self) -> dict:
        return {**{k: self.paths[k].text().strip() for k, _ in self.FOLDERS},
                "session_id": self.session.currentText().strip(), "episodes": self.episode_list(),
                "with_video": self.with_video.isChecked(), "attention": self.attention.isChecked(),
                "overwrite": self.overwrite.isChecked()}


class PathsDialog(QDialog):
    def __init__(self, parent=None):
        from ..config import KEYS, config_path, load_paths
        super().__init__(parent)
        self.setWindowTitle("Default paths")
        self.resize(720, 0)
        form = QFormLayout(self)
        form.addRow(QLabel(f"Saved for this machine only, in\n{config_path()}"))
        cur = load_paths()
        self.edits = {}
        for key, label in KEYS.items():
            e = QLineEdit(cur.get(key, ""))
            b = browse_button()
            b.clicked.connect(lambda _=False, k=key: self._browse(k))
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(e, 1)
            h.addWidget(b)
            form.addRow(label, row)
            self.edits[key] = e
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _browse(self, key):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", self.edits[key].text())
        if d:
            self.edits[key].setText(d)

    def values(self) -> dict:
        return {k: e.text().strip() for k, e in self.edits.items()}
