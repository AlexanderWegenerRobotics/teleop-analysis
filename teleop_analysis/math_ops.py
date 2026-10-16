from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .viewer.common import gap_threshold


@dataclass
class Op:
    key: str
    label: str
    min_inputs: int
    max_inputs: int | None
    each: bool = False
    params: list = field(default_factory=list)
    group: str = "basic"


OPS: dict[str, Op] = {}


def _op(*args, **kw):
    o = Op(*args, **kw)
    OPS[o.key] = o
    return o


_op("se3", "SE(3) → position + RPY + quaternion", 16, 16, group="pose")
_op("quat2rpy", "Quaternion (w, x, y, z) → roll / pitch / yaw", 4, 4, group="pose")
_op("norm", "Magnitude ‖·‖", 1, None)
_op("sum", "Sum a + b + …", 2, None)
_op("diff", "Difference a − b", 2, 2)
_op("mul", "Product a · b", 2, 2)
_op("div", "Ratio a / b", 2, 2)
_op("expr", "Expression…", 1, None, params=[("expression", "Expression in a, b, c, … (numpy allowed)", "a")])
_op("derivative", "Derivative d/dt", 1, None, each=True, group="each")
_op("integral", "Integral ∫ dt", 1, None, each=True, group="each")
_op("movavg", "Moving average…", 1, None, each=True, params=[("window", "Window [s]", 0.05)], group="each")
_op("abs", "Absolute value", 1, None, each=True, group="each")
_op("scale", "Scale and offset…", 1, None, each=True, params=[("gain", "Gain", 1.0), ("offset", "Offset", 0.0)],
    group="each")
_op("shift", "Time shift…", 1, None, each=True, params=[("seconds", "Shift [s] (+ = later)", 0.0)], group="each")
_op("unwrap", "Unwrap angle (rad)", 1, None, each=True, group="each")
_op("rad2deg", "rad → deg", 1, None, each=True, group="each")
_op("deg2rad", "deg → rad", 1, None, each=True, group="each")


def available_ops(n: int) -> list[Op]:
    return [o for o in OPS.values() if n >= o.min_inputs and (o.max_inputs is None or n <= o.max_inputs)]


def resample(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y_src) & np.isfinite(t_src)
    ts, ys = t_src[ok], y_src[ok]
    if len(ts) < 2:
        return np.full(len(t_dst), np.nan)
    if np.array_equal(t_src, t_dst) and ok.all():
        return y_src.astype(np.float64)
    y = np.interp(t_dst, ts, ys, left=np.nan, right=np.nan)
    thr = gap_threshold(ts)
    if np.isfinite(thr):
        i = np.clip(np.searchsorted(ts, t_dst), 1, len(ts) - 1)
        span = ts[i] - ts[i - 1]
        y[span > thr] = np.nan
    return y


def _suffix_int(name: str) -> int:
    m = re.search(r"(\d+)\]?$", name)
    return int(m.group(1)) if m else 0


def order_se3(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: _suffix_int(k.split("::")[-1]))


def order_quat(keys: list[str]) -> list[str]:
    def comp(k):
        c = k.split("::")[-1].lower()
        for i, s in enumerate("wxyz"):
            if re.search(rf"(^|[_.\[]){s}\]?$", c) or re.search(rf"q{s}$", c):
                return i
        return None
    idx = [comp(k) for k in keys]
    if sorted(i for i in idx if i is not None) == [0, 1, 2, 3]:
        return [k for _, k in sorted(zip(idx, keys))]
    return list(keys)


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c0 = tr > 0
    c1 = ~c0 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    c2 = ~c0 & ~c1 & (R[:, 1, 1] > R[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(np.maximum(tr[c0] + 1.0, 0)) * 2
        q[c0] = np.stack([0.25 * s, (R[c0, 2, 1] - R[c0, 1, 2]) / s, (R[c0, 0, 2] - R[c0, 2, 0]) / s,
                          (R[c0, 1, 0] - R[c0, 0, 1]) / s], 1)
        s = np.sqrt(np.maximum(1.0 + R[c1, 0, 0] - R[c1, 1, 1] - R[c1, 2, 2], 0)) * 2
        q[c1] = np.stack([(R[c1, 2, 1] - R[c1, 1, 2]) / s, 0.25 * s, (R[c1, 0, 1] + R[c1, 1, 0]) / s,
                          (R[c1, 0, 2] + R[c1, 2, 0]) / s], 1)
        s = np.sqrt(np.maximum(1.0 + R[c2, 1, 1] - R[c2, 0, 0] - R[c2, 2, 2], 0)) * 2
        q[c2] = np.stack([(R[c2, 0, 2] - R[c2, 2, 0]) / s, (R[c2, 0, 1] + R[c2, 1, 0]) / s, 0.25 * s,
                          (R[c2, 1, 2] + R[c2, 2, 1]) / s], 1)
        s = np.sqrt(np.maximum(1.0 + R[c3, 2, 2] - R[c3, 0, 0] - R[c3, 1, 1], 0)) * 2
        q[c3] = np.stack([(R[c3, 1, 0] - R[c3, 0, 1]) / s, (R[c3, 0, 2] + R[c3, 2, 0]) / s,
                          (R[c3, 1, 2] + R[c3, 2, 1]) / s, 0.25 * s], 1)
    return _continuous(q)


def _continuous(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if len(q) < 2:
        return q
    d = np.einsum("ij,ij->i", q[1:], q[:-1])
    s = np.where(np.nan_to_num(d, nan=1.0) < 0, -1.0, 1.0)
    f = np.concatenate([[1.0], np.cumprod(s)])
    return q * f[:, None]


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def _rpy_deg(R: np.ndarray) -> dict:
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    out = {}
    for n, a in (("roll [deg]", roll), ("pitch [deg]", pitch), ("yaw [deg]", yaw)):
        out[n] = np.degrees(_unwrap_nan(a))
    return out


def _unwrap_nan(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a)
    out = a.astype(np.float64).copy()
    if ok.sum() > 1:
        out[ok] = np.unwrap(a[ok])
    return out


def se3_decompose(M: np.ndarray) -> dict:
    col_major = np.nanmedian(np.abs(M[:, [3, 7, 11]])) < 1e-6 and np.nanmedian(np.abs(M[:, 15] - 1)) < 1e-6
    if col_major:
        R = np.stack([M[:, [0, 4, 8]], M[:, [1, 5, 9]], M[:, [2, 6, 10]]], 1)
        p = M[:, [12, 13, 14]]
    else:
        R = np.stack([M[:, [0, 1, 2]], M[:, [4, 5, 6]], M[:, [8, 9, 10]]], 1)
        p = M[:, [3, 7, 11]]
    out = {"x [m]": p[:, 0], "y [m]": p[:, 1], "z [m]": p[:, 2]}
    out.update(_rpy_deg(R))
    bad = ~np.all(np.isfinite(M), axis=1)
    Rf = np.where(bad[:, None, None], np.eye(3), R)
    q = _quat_from_R(Rf)
    q[bad] = np.nan
    for i, n in enumerate(("qw", "qx", "qy", "qz")):
        out[n] = q[:, i]
    for k in ("roll [deg]", "pitch [deg]", "yaw [deg]"):
        out[k][bad] = np.nan
    return out


def _movavg(t, y, window):
    if window <= 0 or len(t) < 2:
        return y
    s = pd.Series(y, index=pd.to_datetime(t, unit="s"))
    try:
        return s.rolling(pd.Timedelta(seconds=window), center=True, min_periods=1).mean().to_numpy()
    except (ValueError, TypeError):
        dt = float(np.median(np.diff(t))) or 1e-3
        n = max(int(round(window / dt)), 1)
        return s.reset_index(drop=True).rolling(n, center=True, min_periods=1).mean().to_numpy()


def _derivative(t, y):
    out = np.full(len(y), np.nan)
    if len(t) < 3:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.gradient(y, t)
    thr = gap_threshold(t)
    dt = np.diff(t)
    big = np.flatnonzero(dt > thr)
    d[big] = np.nan
    d[np.minimum(big + 1, len(d) - 1)] = np.nan
    return d


def _integral(t, y):
    yy = np.nan_to_num(y, nan=0.0)
    out = np.zeros(len(t))
    if len(t) > 1:
        out[1:] = np.cumsum(0.5 * (yy[1:] + yy[:-1]) * np.diff(t))
    return out


SAFE = {k: getattr(np, k) for k in ("sqrt", "abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
                                     "exp", "log", "log10", "hypot", "degrees", "radians", "sign", "minimum",
                                     "maximum", "clip", "where", "pi", "unwrap", "floor", "ceil", "round")}
SAFE["np"] = np


def evaluate(op: str, ts: list[np.ndarray], ys: list[np.ndarray], names: list[str], params: dict):
    o = OPS[op]
    if op == "shift":
        return ts[0] + float(params.get("seconds", 0.0)), {f"{names[0]}": ys[0]}
    t = ts[0]
    if o.each:
        y = ys[0]
        n = names[0]
        if op == "derivative":
            return t, {f"d/dt {n}": _derivative(t, y)}
        if op == "integral":
            return t, {f"∫ {n} dt": _integral(t, y)}
        if op == "movavg":
            return t, {f"avg {n}": _movavg(t, y, float(params.get("window", 0.05)))}
        if op == "abs":
            return t, {f"|{n}|": np.abs(y)}
        if op == "scale":
            return t, {n: y * float(params.get("gain", 1.0)) + float(params.get("offset", 0.0))}
        if op == "unwrap":
            return t, {f"unwrap {n}": _unwrap_nan(y)}
        if op == "rad2deg":
            return t, {f"{n} [deg]": np.degrees(y)}
        if op == "deg2rad":
            return t, {f"{n} [rad]": np.radians(y)}
    aligned = [ys[0].astype(np.float64)] + [resample(ts[i], ys[i], t) for i in range(1, len(ys))]
    A = np.stack(aligned, 1)
    if op == "se3":
        return t, se3_decompose(A)
    if op == "quat2rpy":
        bad = ~np.all(np.isfinite(A), axis=1)
        R = _R_from_quat(np.where(bad[:, None], [1.0, 0, 0, 0], A))
        out = _rpy_deg(R)
        for v in out.values():
            v[bad] = np.nan
        return t, out
    if op == "norm":
        return t, {"norm": np.sqrt(np.sum(A ** 2, axis=1))}
    if op == "sum":
        return t, {"sum": np.sum(A, axis=1)}
    if op == "diff":
        return t, {"a - b": A[:, 0] - A[:, 1]}
    if op == "mul":
        return t, {"a * b": A[:, 0] * A[:, 1]}
    if op == "div":
        with np.errstate(divide="ignore", invalid="ignore"):
            return t, {"a / b": A[:, 0] / A[:, 1]}
    if op == "expr":
        expr = str(params.get("expression", "a"))
        env = dict(SAFE)
        for i in range(A.shape[1]):
            env[chr(ord("a") + i)] = A[:, i]
        env["t"] = t
        with np.errstate(all="ignore"):
            v = eval(expr, {"__builtins__": {}}, env)
        v = np.broadcast_to(np.asarray(v, dtype=np.float64), t.shape).copy()
        return t, {"expr": v}
    raise ValueError(f"unknown op {op}")


def estimate_delay(t1, y1, t2, y2, window=None, max_lag=1.0) -> tuple[float, float]:
    ok1 = np.isfinite(y1)
    ok2 = np.isfinite(y2)
    t1, y1, t2, y2 = t1[ok1], y1[ok1], t2[ok2], y2[ok2]
    lo = max(t1[0], t2[0])
    hi = min(t1[-1], t2[-1])
    if window:
        lo, hi = max(lo, window[0]), min(hi, window[1])
    if hi - lo <= 0:
        raise ValueError("signals do not overlap in the visible window")
    dt = max(min(float(np.median(np.diff(t1))), float(np.median(np.diff(t2)))), 1e-4)
    n = int((hi - lo) / dt)
    if n > 2_000_000:
        dt = (hi - lo) / 2_000_000
        n = 2_000_000
    if n < 8:
        raise ValueError("window too short")
    g = lo + np.arange(n) * dt
    a = np.interp(g, t1, y1)
    b = np.interp(g, t2, y2)
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(b, m) * np.conj(np.fft.rfft(a, m)), m)
    c = np.concatenate([c[-(n - 1):], c[:n]]) / n
    lags = np.arange(-(n - 1), n) * dt
    sel = np.abs(lags) <= max_lag
    i = int(np.argmax(np.where(sel, c, -np.inf)))
    if 0 < i < len(c) - 1:
        y0, y1_, y2_ = c[i - 1], c[i], c[i + 1]
        den = y0 - 2 * y1_ + y2_
        frac = 0.5 * (y0 - y2_) / den if den != 0 else 0.0
    else:
        frac = 0.0
    return float((i - (n - 1) + frac) * dt), float(c[i])
