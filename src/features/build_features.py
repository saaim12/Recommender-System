from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.io import ensure_dir


def build_train_test(raw_path: Path, processed_dir: Path, test_size: float = 0.2, random_state: int = 42):
    ensure_dir(processed_dir)
    df = pd.read_parquet(raw_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_path, test_path
