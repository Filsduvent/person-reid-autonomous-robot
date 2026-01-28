from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, IO, Optional


def setup_logger(name: str, output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if re-imported
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: Optional[IO[str]] = open(self.path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        if self._f is None:
            raise RuntimeError("JsonlWriter is closed")
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is None:
            return
        try:
            self._f.close()
        finally:
            self._f = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
