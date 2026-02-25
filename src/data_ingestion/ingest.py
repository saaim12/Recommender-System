from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.io import ensure_dir


def ingest_to_parquet(raw_output: Path) -> Path:
    """Placeholder ingestion that creates a tiny sample dataset."""

    ensure_dir(raw_output.parent)
    data = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [101, 102, 101, 103, 104],
            "rating": [5, 4, 3, 5, 2],
        }
    )
    raw_output = raw_output.with_suffix(".parquet")
    data.to_parquet(raw_output, index=False)
    return raw_output
