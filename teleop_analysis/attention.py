"""Offline gaze attention video from an interface episode folder (session.mp4 + frames.jsonl + metadata.json).

Same look as the interface's live attention output: a Gaussian spotlight around the gaze point,
peripheral darkened, so a logged episode without session_attention.mp4 can still be reviewed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_gaze(path: Path) -> list[tuple[float, float, float, bool]]:
    """(u, v, confidence, valid) per logged frame; frames.jsonl objects may span several lines."""
    from .loaders.special import iter_json_objects
    text = Path(path).read_text(encoding="utf-8")
    return [(float(o.get("gaze_u", 0.5)), float(o.get("gaze_v", 0.5)),
             float(o.get("gaze_confidence", 0.0)), bool(o.get("gaze_valid", False))) for o in iter_json_objects(text)]


class _Writer:
    def __init__(self, out: Path, w: int, h: int, fps: float):
        import cv2
        self.proc = None
        self.cv = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            self.proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
                stdin=subprocess.PIPE)
            self.name = "ffmpeg/libx264"
        else:
            self.cv = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            self.name = "opencv/mp4v"

    def write(self, frame) -> None:
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def render_attention(session_dir: Path, out: Path | None = None, radius_frac: float | None = None,
                     dark: float | None = None, dot_radius: int = 6, dot: bool = True, quiet: bool = False) -> Path:
    """Writes session_attention.mp4 next to session.mp4 (or to `out`) and returns its path."""
    import cv2
    d = Path(session_dir)
    video, frames = d / "session.mp4", d / "frames.jsonl"
    if not video.exists() or not frames.exists():
        raise FileNotFoundError(f"missing session.mp4 or frames.jsonl in {d}")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8")) if (d / "metadata.json").exists() else {}
    radius_frac = radius_frac if radius_frac is not None else float(meta.get("gaze_radius_frac", 0.10))
    dark = min(max(dark if dark is not None else float(meta.get("peripheral_brightness", 0.20)), 0.0), 1.0)

    gaze = load_gaze(frames)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("target_fps", 30.0))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = Path(out) if out else d / "session_attention.mp4"
    writer = _Writer(out, w, h, fps)

    sig_sq = (radius_frac * w) ** 2
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    dark_frame = np.float32(dark)
    yy, xx = np.mgrid[-dot_radius:dot_radius + 1, -dot_radius:dot_radius + 1]
    dot_mask = (xx * xx + yy * yy) <= dot_radius * dot_radius
    if not quiet:
        print(f"{video.name}: {w}x{h} @ {fps:.1f} fps, {n_video} frames, {len(gaze)} gaze rows, "
              f"radius_frac={radius_frac} dark={dark} -> {out.name} ({writer.name})")
        if n_video and abs(n_video - len(gaze)) > 2:
            print(f"warning: frame count {n_video} != gaze rows {len(gaze)}; extra frames reuse the last gaze row")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        u, v, conf, valid = gaze[min(i, len(gaze) - 1)] if gaze else (0.5, 0.5, 0.0, False)
        gx, gy = u * w, v * h
        if valid and conf > 0.0:
            ex = np.exp(-0.45 * (xs - gx) ** 2 / (2.0 * sig_sq))
            ey = np.exp(-0.45 * (ys - gy) ** 2 / (2.0 * sig_sq))
            weight = dark_frame + (1.0 - dark_frame) * np.outer(ey, ex)
        else:
            weight = np.full((h, w), dark_frame, dtype=np.float32)
        frame = np.clip(frame.astype(np.float32) * weight[:, :, None], 0, 255).astype(np.uint8)
        if valid and dot:
            cx, cy, r = int(gx), int(gy), dot_radius
            x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
            y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
            if x0 < x1 and y0 < y1:
                m = dot_mask[(y0 - (cy - r)):(y1 - (cy - r)), (x0 - (cx - r)):(x1 - (cx - r))]
                frame[y0:y1, x0:x1][m] = (50, 50, 255)
        writer.write(frame)
        i += 1
        if not quiet and i % 300 == 0:
            print(f"  {i} frames")
    cap.release()
    writer.close()
    if not quiet:
        print(f"done: {i} frames -> {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Render the gaze attention video of an interface episode folder.")
    ap.add_argument("session_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius-frac", type=float, default=None)
    ap.add_argument("--dark", type=float, default=None)
    ap.add_argument("--dot-radius", type=int, default=6)
    ap.add_argument("--no-dot", action="store_true")
    a = ap.parse_args(argv)
    try:
        render_attention(Path(a.session_dir), a.out, a.radius_frac, a.dark, a.dot_radius, not a.no_dot)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
