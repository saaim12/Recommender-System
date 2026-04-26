from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import settings
from src.serving.api import app


if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
