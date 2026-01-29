from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import time

import numpy as np


class UpdateSkipReason(str, Enum):
    LOW_DET_CONF = "low_det_conf"
    LOW_SIMILARITY = "low_similarity"
    SMALL_MARGIN = "small_margin"
    SUSPECT_OCCLUSION = "suspect_occlusion"
    TRACK_UNSTABLE = "track_unstable"
    RECENTLY_REACQUIRED = "recently_reacquired"
    NOT_KNOWN = "not_known"
    NO_EMBEDDING = "no_embedding"


@dataclass
class IdentityRecord:
    identity_id: str
    prototype: np.ndarray
    label: Optional[str] = None
    created_ts: float = field(default_factory=lambda: time.time())
    updated_ts: float = field(default_factory=lambda: time.time())
    num_updates: int = 0
    num_observations: int = 0
    meta: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    best_id: Optional[str]
    best_score: float
    second_score: float
    margin: float
    is_known: bool
