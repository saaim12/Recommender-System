from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(payload: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    Path(path).write_text(pd.io.json.dumps(payload, indent=2), encoding="utf-8")
    return path
