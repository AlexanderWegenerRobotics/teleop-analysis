from __future__ import annotations

import hashlib
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

from .cache import cache_dir

MAX_WIDTH = 1280


def _crop_markers(img: np.ndarray) -> np.ndarray:
    if img.shape[0] % 16 == 2:
        return img[:-2]
    return img


def _shrink(img: np.ndarray) -> np.ndarray:
    if img.shape[1] <= MAX_WIDTH:
        return img
    import cv2
    s = MAX_WIDTH / img.shape[1]
    return cv2.resize(img, (MAX_WIDTH, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def remux_h264(src: Path, progress=None) -> Path | None:
    try:
        import av
    except ImportError:
        return None
    st = src.stat()
    h = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|remux1".encode()).hexdigest()[:20]
    out = cache_dir() / f"{src.stem[:40]}_{h}.mp4"
    if out.exists():
        return out
    tmp = out.with_name(out.stem + ".part.mp4")
    tb = Fraction(1, 30)
    try:
        with av.open(str(src), format="h264") as inp, av.open(str(tmp), "w", format="mp4") as dst:
            ins = inp.streams.video[0]
            try:
                ost = dst.add_stream_from_template(ins)
            except AttributeError:
                ost = dst.add_stream(template=ins)
            ost.time_base = tb
            i = 0
            for pkt in inp.demux(ins):
                if pkt.size == 0:
                    continue
                pkt.pts = i
                pkt.dts = i
                pkt.time_base = tb
                pkt.stream = ost
                dst.mux(pkt)
                i += 1
                if progress and i % 500 == 0:
                    progress(i)
        os.replace(tmp, out)
        return out
    except Exception:
        if tmp.exists():
            tmp.unlink()
        return None


class AvSeeker:
    def __init__(self, path: Path):
        import av
        self._av = av
        self.c = av.open(str(path))
        self.s = self.c.streams.video[0]
        pts, keys = [], []
        for p in self.c.demux(self.s):
            if p.pts is None:
                continue
            pts.append(p.pts)
            if p.is_keyframe:
                keys.append(p.pts)
        if not pts:
            raise IOError("no frames")
        self.pts = np.sort(np.asarray(pts, dtype=np.int64))
        self.keys = np.sort(np.asarray(keys or [self.pts[0]], dtype=np.int64))
        self._gen = None
        self._last_pts = None
        self._cache: dict[int, np.ndarray] = {}

    def count(self) -> int:
        return len(self.pts)

    def frame(self, idx: int) -> np.ndarray | None:
        if idx < 0 or idx >= len(self.pts):
            return None
        target = int(self.pts[idx])
        if target in self._cache:
            return self._cache[target]
        k = int(self.keys[max(int(np.searchsorted(self.keys, target, side="right")) - 1, 0)])
        if self._gen is None or self._last_pts is None or target <= self._last_pts or self._last_pts < k:
            self.c.seek(k, stream=self.s, backward=True, any_frame=False)
            self._gen = self.c.decode(self.s)
            self._last_pts = None
        try:
            for fr in self._gen:
                if fr.pts is None:
                    continue
                self._last_pts = fr.pts
                if fr.pts >= target:
                    img = fr.to_ndarray(format="rgb24")
                    if len(self._cache) > 8:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[target] = img
                    return img
        except self._av.error.EOFError:
            pass
        self._gen = None
        return None

    def close(self):
        self.c.close()


class FrameReader:
    def __init__(self, info: dict, progress=None):
        self.kind = info.get("type")
        self.path = Path(info["path"])
        self._cap = None
        self._pos = -1
        self._h5 = None
        self._ds = None
        self.sequential_only = False
        if self.kind == "h264":
            mp4 = remux_h264(self.path, progress)
            if mp4 is not None:
                self.path = mp4
            else:
                self.sequential_only = True
            self.kind = "cv"
        if self.kind == "h5":
            import h5py
            self._h5 = h5py.File(self.path, "r")
            found = []

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 4 and obj.dtype == np.uint8:
                    found.append(obj)
            self._h5.visititems(visit)
            if not found:
                raise ValueError(f"no image dataset in {self.path.name}")
            self._ds = found[0]
        else:
            self._av = None
            try:
                self._av = AvSeeker(self.path)
            except Exception:
                self._av = None
            if self._av is None:
                self._open()

    def _open(self):
        import cv2
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        self._pos = -1
        if not self._cap.isOpened():
            raise IOError(f"cannot open {self.path.name}")

    def count(self) -> int:
        if self._ds is not None:
            return int(self._ds.shape[0])
        if getattr(self, "_av", None) is not None:
            return self._av.count()
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frame(self, idx: int) -> np.ndarray | None:
        idx = int(idx)
        if idx < 0:
            return None
        if self._ds is not None:
            if idx >= self._ds.shape[0]:
                return None
            return _shrink(np.ascontiguousarray(self._ds[idx]))
        if getattr(self, "_av", None) is not None:
            img = self._av.frame(idx)
            return None if img is None else _shrink(_crop_markers(img))
        import cv2
        if idx == self._pos and getattr(self, "_last", None) is not None:
            return self._last
        if idx <= self._pos or idx - self._pos > 90:
            if self.sequential_only:
                if idx <= self._pos:
                    self._open()
            else:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._pos = idx - 1
        while self._pos < idx - 1:
            if not self._cap.grab():
                return None
            self._pos += 1
        ok, img = self._cap.read()
        if not ok or img is None:
            return None
        self._pos += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._last = _shrink(_crop_markers(img))
        return self._last

    def close(self):
        if getattr(self, "_av", None) is not None:
            self._av.close()
            self._av = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
