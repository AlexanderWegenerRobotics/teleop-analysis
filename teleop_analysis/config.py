"""Machine-specific default paths (log roots, dataset folder, sibling repos), kept outside the repo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KEYS = {
    "avatar": "Avatar / simulator log root (also holds camera_params.json)",
    "local": "Local orchestrator log root",
    "interface": "VR interface log root",
    "out": "Dataset folder (collected episodes)",
    "orchestrator": "teleop-orchestrator repo (task goals for intent names)",
}


def config_path() -> Path:
    env = os.environ.get("TELEOP_ANALYSIS_CONFIG")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "teleop-analysis" / "paths.json"


def _legacy() -> dict:
    p = Path(__file__).with_name("collect_defaults.json")
    try:
        return {k: v for k, v in json.loads(p.read_text()).items() if k in KEYS and v}
    except (OSError, ValueError):
        return {}


def load_paths() -> dict[str, str]:
    out = {k: "" for k in KEYS}
    out.update(_legacy())
    try:
        out.update({k: v for k, v in json.loads(config_path().read_text()).items() if k in KEYS})
    except (OSError, ValueError):
        pass
    return out


def save_paths(paths: dict[str, str]) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {k: str(paths.get(k, "") or "") for k in KEYS}
    p.write_text(json.dumps(data, indent=2))
    return p


def path(key: str) -> Path | None:
    v = load_paths().get(key) or ""
    return Path(v) if v else None
