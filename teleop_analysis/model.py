from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SIGNALS = "signals"
EVENTS = "events"
VIDEO = "video"

CLOCK_INTERFACE = "interface"
CLOCK_OPERATOR = "operator"
CLOCK_AVATAR = "avatar"
CLOCK_RELATIVE = "relative"

CLOCK_LABELS = {
    CLOCK_INTERFACE: "interface / operator PC (reference)",
    CLOCK_OPERATOR: "operator camera PC",
    CLOCK_AVATAR: "avatar / NUC",
    CLOCK_RELATIVE: "relative time (no absolute clock)",
}

TIME_COL = "__t"
SPLIT_COL = "__split"


@dataclass
class Stream:
    root: "Root"
    rel: str
    name: str
    kind: str
    clock: str
    loader: object
    columns: list[str] = field(default_factory=list)
    split_value: object = None
    sub: str | None = None
    t_first_ns: int | None = None
    deps: list[Path] = field(default_factory=list)
    video: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    error: str | None = None
    empty: bool = False

    @property
    def path(self) -> Path:
        return self.root.path / self.rel

    @property
    def folder(self) -> str:
        p = str(PurePosixPath(self.rel).parent)
        return "" if p == "." else p

    @property
    def key(self) -> str:
        return f"{self.root.name}/{self.rel}" + ("" if self.sub is None else f"#{self.sub}")

    @property
    def label(self) -> str:
        parts = [p for p in (self.folder, self.name) if p]
        return "/".join(parts)

    def signal_key(self, column: str) -> str:
        return f"{self.key}::{column}"


@dataclass
class Root:
    path: Path
    name: str
    streams: list[Stream] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def by_key(self) -> dict[str, Stream]:
        return {s.key: s for s in self.streams}


def split_signal_key(key: str) -> tuple[str, str | None]:
    if "::" in key:
        a, b = key.split("::", 1)
        return a, b
    return key, None
