"""Pulls one episode from the three producers into a dataset folder.

Every producer writes under the avatar's session_id / episode index (teleop-simulator EpisodeEventMsg):
  avatar       <avatar_logs>/<NNN>/                 arm_*.csv, scene.csv, intention_log.csv, video_*.h264, episode.hdf5 ...
  orchestrator <local_logs>/<session_id>/<NNN>/     orchestrator.hdf5, nudge.csv
  interface    <interface_logs>/<session_id>/<NNN>/ stream.csv, command.csv, events.log, frames.jsonl, metadata.json, session*.mp4

Result: <out>/<NNN>/ with the avatar files and orchestrator.hdf5 / nudge.csv at the top and the interface files
under interface/ (so the viewer puts them on the interface clock). Videos only with --with-video.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

INTERFACE_FILES = ("stream.csv", "command.csv", "events.log", "frames.jsonl", "metadata.json")
INTERFACE_VIDEO = ("session.mp4", "session_attention.mp4")
LOCAL_FILES = ("orchestrator.hdf5", "local.hdf5", "nudge.csv")
AVATAR_VIDEO_SUFFIXES = (".h264", ".mp4", ".mkv")


def resolve_path(path: str | Path) -> tuple[Path, str | None, int | None]:
    """Accepts a log root, a <session_id>/ folder or a <NNN>/ episode folder and returns (root, session_id, episode)."""
    p = Path(path)
    session, episode = None, None
    if p.name.isdigit() and len(p.name) <= 4:
        episode = int(p.name)
        p = p.parent
    if p.name.isdigit() and len(p.name) > 4:
        session = p.name
        p = p.parent
    if session is None and episode is not None:
        session = avatar_session_id(Path(path))
    return p, session, episode


def sibling_defaults(path: str | Path) -> dict[str, str]:
    """Guesses the other log roots next to a known one (repos checked out side by side)."""
    root = str(resolve_path(path)[0]).replace("\\", "/")
    out = {}
    for key, repo in (("avatar", "teleop-simulator"), ("local", "teleop-orchestrator")):
        for other, other_repo in (("avatar", "teleop-simulator"), ("local", "teleop-orchestrator")):
            if other != key and f"/{other_repo}/" in root:
                cand = Path(root.replace(f"/{other_repo}/", f"/{repo}/"))
                if cand.is_dir():
                    out[key] = str(cand)
    return out


def avatar_session_id(folder: Path) -> str | None:
    """session_id from the first *_meta.csv in an avatar episode folder, None if there is none."""
    for meta in sorted(folder.glob("*_meta.csv")):
        with open(meta, newline="") as f:
            row = next(csv.DictReader(f, delimiter=";"), None)
        if row and row.get("session_id"):
            return row["session_id"]
    return None


def _copy(src: Path, dst: Path, base: Path, copied: list[str], overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst.relative_to(base).as_posix())


def collect_episode(session_id: str, episode: int, out: Path, avatar_logs: Path | None, local_logs: Path | None,
                    interface_logs: Path | None, with_video: bool = False, attention: bool = False,
                    out_index: int | None = None, overwrite: bool = False) -> tuple[Path, list[str], list[str]]:
    """Copies the episode into out/<index>/; returns (folder, copied files, warnings)."""
    idx = f"{episode:03d}"
    dst = out / f"{(out_index if out_index is not None else episode):03d}"
    copied: list[str] = []
    warnings: list[str] = []

    if avatar_logs is not None:
        src = avatar_logs / idx
        if not src.is_dir():
            warnings.append(f"avatar: no folder {src}")
        else:
            sid = avatar_session_id(src)
            if sid and sid != session_id:
                warnings.append(f"avatar: {src} belongs to session {sid}, not {session_id}")
            for f in sorted(src.iterdir()):
                if f.is_dir() or (f.suffix in AVATAR_VIDEO_SUFFIXES and not with_video):
                    continue
                _copy(f, dst / f.name, dst, copied, overwrite)
            cam = avatar_logs / "camera_params.json"
            if cam.exists() and not (src / "camera_params.json").exists():
                _copy(cam, dst / "camera_params.json", dst, copied, overwrite)

    if local_logs is not None:
        src = local_logs / session_id / idx
        if not src.is_dir():
            warnings.append(f"orchestrator: no folder {src}")
        for name in LOCAL_FILES:
            if (src / name).exists():
                _copy(src / name, dst / name, dst, copied, overwrite)

    if interface_logs is not None:
        src = interface_logs / session_id / idx
        if not src.is_dir():
            warnings.append(f"interface: no folder {src}")
        else:
            names = INTERFACE_FILES + (INTERFACE_VIDEO if with_video else ())
            for name in names:
                if (src / name).exists():
                    _copy(src / name, dst / "interface" / name, dst, copied, overwrite)
            if with_video and attention and not (dst / "interface" / "session_attention.mp4").exists():
                from .attention import render_attention
                try:
                    render_attention(dst / "interface", quiet=True)
                    copied.append("interface/session_attention.mp4")
                except (FileNotFoundError, RuntimeError) as e:
                    warnings.append(f"attention: {e}")
    return dst, copied, warnings


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", help="avatar session id (EpisodeEventMsg.session_id, also in *_meta.csv)")
    ap.add_argument("episode", type=int, nargs="+", help="episode index/indices")
    ap.add_argument("--out", required=True, help="dataset folder, episodes land in <out>/<NNN>/")
    ap.add_argument("--avatar", type=Path, help="avatar log_dir (holds <NNN>/)")
    ap.add_argument("--local", type=Path, help="orchestrator logging.dir (holds <session_id>/<NNN>/)")
    ap.add_argument("--interface", type=Path, help="interface Saved/Logs/TeleOp (holds <session_id>/<NNN>/)")
    ap.add_argument("--with-video", action="store_true", help="also copy the avatar camera and interface videos")
    ap.add_argument("--attention", action="store_true", help="with --with-video: render session_attention.mp4 if missing")
    ap.add_argument("--out-index", type=int, help="index in <out> for a single episode (default: the avatar index)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not (a.avatar or a.local or a.interface):
        sys.exit("give at least one of --avatar, --local, --interface")
    if a.out_index is not None and len(a.episode) != 1:
        sys.exit("--out-index only with a single episode")
    for ep in a.episode:
        dst, copied, warnings = collect_episode(a.session_id, ep, Path(a.out), a.avatar, a.local, a.interface,
                                               a.with_video, a.attention, a.out_index, a.overwrite)
        print(f"{dst}: {len(copied)} files")
        for w in warnings:
            print(f"  warning: {w}")


if __name__ == "__main__":
    main()
