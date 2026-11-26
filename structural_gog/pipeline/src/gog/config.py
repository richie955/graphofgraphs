"""Config loading. Import `cfg()` everywhere instead of reading config.yaml directly."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def cfg(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(p) as fh:
        conf = yaml.safe_load(fh)
    # Resolve every path relative to the repo root so scripts work from any cwd.
    conf["paths"] = {k: (REPO_ROOT / v) for k, v in conf["paths"].items()}
    for key in ("cache_dir", "interim_dir", "out_dir"):
        conf["paths"][key].mkdir(parents=True, exist_ok=True)
    return conf
