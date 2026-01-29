from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import time

import numpy as np

from edge_reid_runtime.gallery.types import IdentityRecord, MatchResult


@dataclass(frozen=True)
class GalleryConfig:
    known_threshold: float = 0.55
    unknown_threshold: float = 0.45
    ema_alpha: float = 0.1
    topk: int = 5
    max_identities: int = 500
    id_prefix: str = "person_"
    id_width: int = 4


class GalleryManager:
    def __init__(self, cfg: Optional[GalleryConfig] = None):
        self.cfg = cfg or GalleryConfig()
        self._entries: Dict[str, IdentityRecord] = {}
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._entries)

    def _new_identity_id(self) -> str:
        identity_id = f"{self.cfg.id_prefix}{self._next_id:0{self.cfg.id_width}d}"
        self._next_id += 1
        return identity_id

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def add(self, identity: Optional[str], embedding: np.ndarray, ts: float, meta: Optional[Dict[str, Any]] = None) -> str:
        if len(self._entries) >= self.cfg.max_identities:
            raise RuntimeError(f"Gallery is full (max_identities={self.cfg.max_identities}).")
        emb = self._l2_normalize(embedding.astype(np.float32))
        if identity is None:
            identity = self._new_identity_id()
        entry = IdentityRecord(
            identity_id=identity,
            prototype=emb,
            label=None,
            created_ts=ts,
            updated_ts=ts,
            num_updates=0,
            num_observations=1,
            meta=meta or {},
        )
        self._entries[identity] = entry
        return identity

    def update(self, identity: str, embedding: np.ndarray, ts: float) -> None:
        if identity not in self._entries:
            return
        entry = self._entries[identity]
        emb = self._l2_normalize(embedding.astype(np.float32))
        entry.prototype = (1.0 - self.cfg.ema_alpha) * entry.prototype + self.cfg.ema_alpha * emb
        entry.prototype = self._l2_normalize(entry.prototype)
        entry.updated_ts = ts
        entry.num_updates += 1
        entry.num_observations += 1

    def search(self, embedding: np.ndarray, topk: Optional[int] = None) -> List[Tuple[str, float]]:
        if not self._entries:
            return []
        emb = self._l2_normalize(embedding.astype(np.float32))
        topk = topk or self.cfg.topk
        scores: List[Tuple[str, float]] = []
        for identity, entry in self._entries.items():
            score = float(np.dot(emb, entry.prototype))
            scores.append((identity, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self._entries:
            return MatchResult(best_id=None, best_score=-1.0, second_score=-1.0, margin=0.0, is_known=False)
        scores = self.search(embedding, topk=2)
        best_id, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else -1.0
        margin = best_score - second_score if second_score > -1.0 else best_score
        is_known = best_score >= self.cfg.known_threshold
        return MatchResult(
            best_id=best_id,
            best_score=float(best_score),
            second_score=float(second_score),
            margin=float(margin),
            is_known=is_known,
        )

    def get_entry(self, identity: str) -> Optional[IdentityRecord]:
        return self._entries.get(identity)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": {
                "known_threshold": self.cfg.known_threshold,
                "unknown_threshold": self.cfg.unknown_threshold,
                "ema_alpha": self.cfg.ema_alpha,
                "topk": self.cfg.topk,
                "max_identities": self.cfg.max_identities,
                "id_prefix": self.cfg.id_prefix,
                "id_width": self.cfg.id_width,
            },
            "entries": [],
        }
        for entry in self._entries.values():
            data["entries"].append(
                {
                    "identity_id": entry.identity_id,
                    "prototype": entry.prototype.tolist(),
                    "label": entry.label,
                    "created_ts": entry.created_ts,
                    "updated_ts": entry.updated_ts,
                    "num_updates": entry.num_updates,
                    "num_observations": entry.num_observations,
                    "meta": entry.meta,
                }
            )
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("entries", [])
        self._entries = {}
        for e in entries:
            rec = IdentityRecord(
                identity_id=e["identity_id"],
                prototype=np.asarray(e["prototype"], dtype=np.float32),
                label=e.get("label"),
                created_ts=float(e.get("created_ts", time.time())),
                updated_ts=float(e.get("updated_ts", time.time())),
                num_updates=int(e.get("num_updates", 0)),
                num_observations=int(e.get("num_observations", 0)),
                meta=e.get("meta", {}) or {},
            )
            self._entries[rec.identity_id] = rec
        # ensure new identities don't overwrite old ones
        if self._entries:
            max_id = 0
            for key in self._entries.keys():
                if key.startswith(self.cfg.id_prefix):
                    suffix = key[len(self.cfg.id_prefix):]
                    try:
                        max_id = max(max_id, int(suffix))
                    except ValueError:
                        continue
            self._next_id = max_id + 1 if max_id > 0 else 1
