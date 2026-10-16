from __future__ import annotations

import os
from pathlib import Path

from ..model import Root
from .base import Loader
from .csv_table import CsvLoader
from .local_run import LocalHdf5Loader, NudgeLoader
from .rosbag import RosbagLoader
from .special import EventsLogLoader, FramesJsonlLoader, MetaLoader, VideoTimestampsLoader, WrenchTruthLoader

LOADERS: list[Loader] = [
    LocalHdf5Loader(),
    NudgeLoader(),
    MetaLoader(),
    VideoTimestampsLoader(),
    WrenchTruthLoader(),
    EventsLogLoader(),
    FramesJsonlLoader(),
    RosbagLoader(),
    CsvLoader(),
]

IGNORED_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".h264", ".hdf5", ".h5", ".json", ".py", ".pyc",
                    ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".yaml", ".yml", ".md", ".txt", ".db3-shm",
                    ".db3-wal", ".parquet", ".zip"}
IGNORED_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".venv", "venv"}


def register(loader: Loader, first: bool = True) -> None:
    if first:
        LOADERS.insert(0, loader)
    else:
        LOADERS.insert(len(LOADERS) - 1, loader)


def loader_for(path: Path) -> Loader | None:
    for ld in LOADERS:
        try:
            if ld.match(path):
                return ld
        except Exception:
            continue
    return None


def discover_root(path: Path, name: str | None = None) -> Root:
    path = Path(path)
    root = Root(path=path, name=name or path.name)
    if path.is_file():
        files = [path]
        root.path = path.parent
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                files.append(Path(dirpath) / fn)
    for f in files:
        rel = f.relative_to(root.path).as_posix()
        ld = loader_for(f)
        if ld is None:
            if f.suffix.lower() not in IGNORED_SUFFIXES and not f.name.endswith(".ffmpeg.log"):
                root.skipped.append(rel)
            continue
        try:
            root.streams.extend(ld.discover(root, f, rel))
        except Exception as e:
            root.errors.append(f"{rel}: {type(e).__name__}: {e}")
    for s in root.streams:
        res = getattr(s.loader, "resolve_clock", None)
        if res:
            res(s)
    return root
