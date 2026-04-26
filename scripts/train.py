from __future__ import annotations

import json

from src.training.train import train_recommender


if __name__ == "__main__":
    print(json.dumps(train_recommender(), indent=2))
