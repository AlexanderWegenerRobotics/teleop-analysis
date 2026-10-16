from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import split_signal_key

MAX_PLOT_POINTS = 20000


def _clip(t, y, a, b):
    i0 = max(int(np.searchsorted(t, a, side="left")) - 1, 0)
    i1 = min(int(np.searchsorted(t, b, side="right")) + 1, len(t))
    return t[i0:i1], y[i0:i1]


def _decimate(t, y, buckets=4000):
    if len(t) <= MAX_PLOT_POINTS:
        return t, y
    edges = np.linspace(0, len(t), buckets + 1).astype(int)
    tt, yy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        if np.all(np.isnan(seg)):
            tt += [t[a]]
            yy += [np.nan]
            continue
        i, j = np.nanargmin(seg), np.nanargmax(seg)
        for k in sorted((i, j)):
            tt.append(t[a + k])
            yy.append(seg[k])
    return np.asarray(tt), np.asarray(yy)


def _gaps(t, y):
    from .viewer.common import with_gaps
    return with_gaps(t, y)


def _unique_labels(signals):
    seen = {}
    out = []
    for s in signals:
        lab = s.get("label") or split_signal_key(s["key"])[1] or s["key"]
        if lab in seen:
            seen[lab] += 1
            lab = f"{lab} ({seen[lab]})"
        else:
            seen[lab] = 1
        out.append(lab)
    return out


def export_view(catalog, view: dict, base, formats=("png",), width_cm=16.0, panel_height_cm=4.5,
                dpi=200, events=True, data=False, title="", view_json=False, frames=None,
                frames_mode="inside", cursors=None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    a, b = view["x_range"]
    panels = [p for p in view["panels"] if p.get("signals")]
    frames = [f for f in (frames or []) if f[1] is not None]
    written = []
    n = max(len(panels), 1)
    inside = bool(frames) and frames_mode == "inside"
    if formats:
        w = width_cm * (1.45 if inside else 1.0)
        fig = plt.figure(figsize=(w / 2.54, panel_height_cm * n / 2.54), layout="constrained")
        if inside:
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.35])
            left = gs[0, 0].subgridspec(n, 1)
            right = gs[0, 1].subgridspec(len(frames), 1)
        else:
            left = fig.add_gridspec(n, 1)
        axes = []
        for i in range(n):
            axes.append(fig.add_subplot(left[i, 0], sharex=axes[0] if axes else None))
        for ax, p in zip(axes, panels):
            labels = _unique_labels(p["signals"])
            for s, lab in zip(p["signals"], labels):
                try:
                    t, y = catalog.get(s["key"])
                except Exception:
                    continue
                t, y = _clip(t, y, a, b)
                tg, yg = _gaps(t, y)
                tg, yg = _decimate(tg, yg)
                ls = {"dash": "--", "dot": ":", "dashdot": "-."}.get(s.get("style"), "-")
                style = dict(color=s.get("color"), lw=0.9 * float(s.get("width") or 1.3) / 1.3, label=lab, ls=ls)
                if len(t) < 300:
                    style.update(marker="o", ms=2)
                ax.plot(tg, yg, **style)
            if p.get("y_range"):
                ax.set_ylim(*p["y_range"])
            if p.get("y_label"):
                ax.set_ylabel(p["y_label"])
            ax.grid(True, alpha=0.3, lw=0.5)
            ax.legend(loc="upper right", fontsize=7, frameon=False)
            if events:
                for ev in view.get("event_list", []):
                    if a <= ev[0] <= b:
                        ax.axvline(ev[0], color="0.5", lw=0.6, ls="--", zorder=0)
            for ct, col, _ in cursors or []:
                if a <= ct <= b:
                    ax.axvline(ct, color=col, lw=1.0, zorder=5)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        if events and panels:
            from .viewer.common import merge_events
            shown = merge_events([tuple(ev) for ev in view.get("event_list", []) if a <= ev[0] <= b],
                                 tol=(b - a) / 150)
            if len(shown) <= 40:
                for ev in shown:
                    axes[0].annotate(ev[1], xy=(ev[0], 1.0), xycoords=("data", "axes fraction"), rotation=90,
                                     fontsize=6, color="0.35", ha="right", va="top")
        if cursors:
            for ct, col, lab in cursors:
                if a <= ct <= b and lab:
                    axes[0].annotate(lab, xy=(ct, 1.0), xycoords=("data", "axes fraction"), fontsize=7,
                                     color=col, ha="left", va="bottom")
        axes[-1].set_xlim(a, b)
        axes[-1].set_xlabel("time [s]")
        if title:
            axes[0].set_title(title, fontsize=9)
        if inside:
            for i, (name, img, tf) in enumerate(frames):
                ax = fig.add_subplot(right[i, 0])
                ax.imshow(img)
                ax.set_axis_off()
                ax.set_title(f"{name}  t = {tf:.3f} s", fontsize=7)
        for fmt in formats:
            out = base.with_suffix(f".{fmt}")
            fig.savefig(out, dpi=dpi)
            written.append(out)
        plt.close(fig)
    if frames and (frames_mode != "inside" or not formats):
        for i, (name, img, tf) in enumerate(frames):
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40]
            out = base.with_name(f"{base.name}_frame{i + 1}_{safe}.png")
            plt.imsave(out, img)
            written.append(out)
    if data:
        rows = []
        sigs = [s for p in panels for s in p["signals"]]
        for s, lab in zip(sigs, _unique_labels(sigs)):
            try:
                t, y = catalog.get(s["key"])
            except Exception:
                continue
            m = (t >= a) & (t <= b)
            rows.append(pd.DataFrame({"signal": lab, "key": s["key"], "t_s": t[m], "value": y[m]}))
        if rows:
            long = pd.concat(rows, ignore_index=True)
            out = base.with_name(base.name + "_data.csv")
            long.to_csv(out, index=False)
            written.append(out)
            wide = None
            for lab, g in long.groupby("signal", sort=False):
                col = g.drop_duplicates("t_s").set_index("t_s")["value"].rename(lab)
                wide = col.to_frame() if wide is None else wide.join(col, how="outer")
            out = base.with_name(base.name + "_wide.csv")
            wide.sort_index().to_csv(out)
            written.append(out)
    if view_json:
        out = base.with_name(base.name + "_view.json")
        out.write_text(json.dumps(view, indent=2))
        written.append(out)
    return written
