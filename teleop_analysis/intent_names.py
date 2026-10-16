"""Human-readable names for the intent module's posteriors in local.hdf5 (phases, candidates) and the camera model
needed to draw them into a video."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

PHASE_NAMES = ["idle", "approach", "grasp", "transport", "place", "insert", "correct", "hold", "handover"]
PHASE_COLORS = ["#a0a0a0", "#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#edc948", "#76b7b2", "#9c755f"]
CAND_OBJECT, CAND_BIN, CAND_GOAL = 0, 1, 2
POSTERIORS = {"phase_posterior": "phase", "target_posterior": "target", "location_posterior": "location"}


def _decode(v) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return str(v).strip("\x00 ")


def phase_names(f) -> list[str]:
    for holder in (f.get("modules/intent"), f):
        if holder is not None and "phase_names" in holder.attrs:
            return [_decode(v).lower() for v in holder.attrs["phase_names"]]
    return list(PHASE_NAMES)


def _dominant_row(arr: np.ndarray):
    rows = [tuple(r) for r in arr]
    counts = Counter(r for r in rows if any(r))
    return counts.most_common(1)[0][0] if counts else (rows[0] if rows else ())


def _slot_positions(world_pos: np.ndarray, n: int = 50) -> np.ndarray:
    head = world_pos[: max(n, 1)]
    out = np.full(head.shape[1:], np.nan)
    ok = np.isfinite(head).any(axis=0)
    if ok.any():
        out[ok] = np.nanmedian(head[:, ok], axis=0)
    return out


def _scene_objects(folder: Path) -> list[tuple[str, np.ndarray]]:
    path = folder / "scene.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh, delimiter=";")
        try:
            head = next(r)
            row = next(r)
        except StopIteration:
            return []
    col = {c: i for i, c in enumerate(head)}
    out = []
    for i in range(64):
        for kind in ("obj", "bin"):
            n = f"{kind}{i}_name"
            if n not in col:
                continue
            try:
                name = row[col[n]].strip()
                p = np.array([float(row[col[f"{kind}{i}_{a}"]]) for a in "xyz"])
            except (KeyError, ValueError, IndexError):
                continue
            if name:
                out.append((name, p))
    return out


def _find_orchestrator(start: Path) -> list[Path]:
    import os
    cands = [Path(os.environ["TELEOP_ORCHESTRATOR"])] if os.environ.get("TELEOP_ORCHESTRATOR") else []
    here = Path(__file__).resolve().parent.parent
    cands.append(here.parent / "teleop-orchestrator")
    from .config import load_paths
    d = load_paths()
    if d.get("orchestrator"):
        cands.insert(0, Path(d["orchestrator"]))
    if d.get("local"):
        cands.append(Path(d["local"]).parent)
    for p in [start, *start.parents]:
        cands.append(p / "teleop-orchestrator")
    return [c for c in cands if (c / "configs").is_dir()]


def _goals_from_config(f, folder: Path) -> dict[str, np.ndarray]:
    cfg = _decode(f.attrs.get("config_path", "")).replace("\\", "/")
    if not cfg:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    for repo in _find_orchestrator(folder):
        p = repo / cfg
        if not p.exists():
            continue
        try:
            conf = yaml.safe_load(p.read_text()) or {}
            rel = (conf.get("geometry") or {}).get("goals_from")
            if not rel:
                return {}
            for base in (repo, p.parent):
                task = base / rel
                if task.exists():
                    goals = (yaml.safe_load(task.read_text()) or {}).get("goals") or {}
                    return {k: np.asarray(v["pos"], dtype=float) for k, v in goals.items() if "pos" in v}
        except Exception:
            return {}
    return {}


def _match(names: list[str], slots: list[int], pos: np.ndarray, known: list[tuple[str, np.ndarray]], tol: float):
    used = set()
    for i in slots:
        if not np.all(np.isfinite(pos[i])) or not known:
            continue
        d = [np.linalg.norm(pos[i] - p) if n not in used else np.inf for n, p in known]
        j = int(np.argmin(d))
        if d[j] <= tol:
            names[i] = known[j][0]
            used.add(known[j][0])


def candidate_names(f, folder: Path) -> tuple[list[str], list[int], str]:
    """(name per candidate slot, type per slot, how the names were found)."""
    if "frames/candidate_types" not in f:
        return [], [], "none"
    types = list(_dominant_row(np.asarray(f["frames/candidate_types"][:])))
    n = len(types)
    mask = list(_dominant_row(np.asarray(f["frames/candidate_mask"][:]))) if "frames/candidate_mask" in f else [True] * n
    if "frames/candidate_names" in f:
        raw = f["frames/candidate_names"][:]
        rows = [tuple(_decode(v) for v in r) for r in raw]
        dom = _dominant_row(np.array(rows, dtype=object)) if rows else ()
        if dom:
            return [x for x in dom], types, "logged"
    names = [""] * n
    source = "types"
    if "frames/candidate_world_pos" in f:
        pos = _slot_positions(np.asarray(f["frames/candidate_world_pos"][:], dtype=float))
        objs = [i for i in range(n) if mask[i] and types[i] in (CAND_OBJECT, CAND_BIN)]
        goals = [i for i in range(n) if mask[i] and types[i] == CAND_GOAL]
        scene = _scene_objects(folder)
        if scene and objs:
            _match(names, objs, pos, scene, 0.06)
            source = "scene.csv"
        task_goals = list(_goals_from_config(f, folder).items())
        if task_goals and goals:
            _match(names, goals, pos, task_goals, 0.01)
            source = "scene.csv + task goals" if source == "scene.csv" else "task goals"
    k_obj = k_goal = 0
    for i in range(n):
        if names[i]:
            continue
        if not mask[i]:
            names[i] = ""
        elif types[i] == CAND_GOAL:
            names[i] = f"goal {k_goal}"
            k_goal += 1
        else:
            names[i] = f"object {k_obj}"
            k_obj += 1
    return names, types, source


def column_labels(f, folder: Path, group: str, columns: list[str]) -> tuple[dict, dict, str]:
    """Labels and colours for the posterior columns of one modules/<intent>/<arm> group."""
    labels, colors = {}, {}
    if not any(c.startswith(tuple(POSTERIORS)) for c in columns):
        return labels, colors, ""
    phases = phase_names(f)
    cands, types, source = candidate_names(f, folder)
    for c in columns:
        for prefix, kind in POSTERIORS.items():
            if not c.startswith(prefix + "_"):
                continue
            try:
                k = int(c[len(prefix) + 1:])
            except ValueError:
                continue
            if kind == "phase":
                name = phases[k] if k < len(phases) else f"phase {k}"
                labels[c] = f"phase {name}"
                if k < len(PHASE_COLORS):
                    colors[c] = PHASE_COLORS[k]
            else:
                name = cands[k] if k < len(cands) and cands[k] else f"slot {k} (empty)"
                if k < len(types) and types[k] == CAND_GOAL and cands[k] and not name.startswith("goal"):
                    name = f"goal {name}"
                labels[c] = f"{kind} {name}"
    return labels, colors, source


def find_camera_params(folder: Path, extra: list[Path] | None = None) -> dict:
    cands = []
    for p in [folder, *folder.parents]:
        cands.append(p / "camera_params.json")
    for e in extra or []:
        cands.append(Path(e) / "camera_params.json" if Path(e).is_dir() else Path(e))
    from .config import load_paths
    d = load_paths()
    if d.get("avatar"):
        cands.append(Path(d["avatar"]) / "camera_params.json")
    for c in cands:
        try:
            if c.exists():
                return json.loads(c.read_text())
        except (OSError, ValueError):
            continue
    return {}


def camera_for_video(params: dict, video_name: str) -> dict | None:
    stem = video_name.lower()
    for k in ("video_", ".h264", ".mp4", ".timestamps.csv"):
        stem = stem.replace(k, "")
    for name, c in params.items():
        n = name.lower()
        if n == stem or n in stem or stem in n:
            return c
    return None
