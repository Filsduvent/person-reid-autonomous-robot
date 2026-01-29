from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from edge_reid_runtime.core.interfaces import Track
from edge_reid_runtime.gallery.manager import GalleryConfig, GalleryManager
from edge_reid_runtime.gallery.types import UpdateSkipReason, MatchResult


@dataclass(frozen=True)
class AssignerConfig:
    stable_age: int = 10
    stable_hits: int = 5
    unknown_threshold: float = 0.45
    known_threshold: float = 0.55
    update_threshold: float = 0.65
    margin_threshold: float = 0.15
    min_det_conf: float = 0.35
    area_drop_ratio: float = 0.5
    aspect_ratio_min: float = 0.2
    aspect_ratio_max: float = 0.9
    reacquire_cooldown_frames: int = 15


@dataclass
class Assignment:
    track_id: int
    identity_id: str
    label: str
    score: float
    margin: float
    enrolled: bool = False
    updated: bool = False
    skip_reason: Optional[str] = None


@dataclass
class _TrackState:
    last_area: Optional[float] = None
    last_aspect: Optional[float] = None
    area_hist: list[float] = None
    age: int = 0
    hits: int = 0
    consecutive_hits: int = 0
    last_seen_frame: int = -1
    reacquired_frame: int = -10**9

    def __post_init__(self):
        if self.area_hist is None:
            self.area_hist = []


class IdentityAssigner:
    def __init__(self, gallery: GalleryManager, cfg: Optional[AssignerConfig] = None):
        self.gallery = gallery
        self.cfg = cfg or AssignerConfig()
        self._track_state: Dict[int, _TrackState] = {}

    @staticmethod
    def _bbox_area_aspect(track: Track) -> Tuple[float, float]:
        x1, y1, x2, y2 = track.bbox_xyxy
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        area = w * h
        aspect = w / h
        return area, aspect

    def _track_stable(self, state: _TrackState) -> bool:
        return state.age >= self.cfg.stable_age and state.consecutive_hits >= self.cfg.stable_hits

    def mark_missed_tracks(self, frame_id: int, active_ids: set[int]) -> None:
        for tid, st in self._track_state.items():
            if tid not in active_ids and st.last_seen_frame == frame_id - 1:
                st.consecutive_hits = 0

    def _step_track_state(self, frame_id: int, track: Track) -> _TrackState:
        tid = int(track.track_id)
        st = self._track_state.setdefault(tid, _TrackState())
        st.age += 1
        st.hits += 1
        st.consecutive_hits += 1
        if st.last_seen_frame < frame_id - 1:
            st.reacquired_frame = frame_id
        st.last_seen_frame = frame_id
        area, aspect = self._bbox_area_aspect(track)
        st.last_area = area
        st.last_aspect = aspect
        st.area_hist.append(area)
        if len(st.area_hist) > 30:
            st.area_hist = st.area_hist[-30:]
        return st

    def _should_update(self, st: _TrackState, track: Track, score: float, margin: float) -> Optional[UpdateSkipReason]:
        if track.conf < self.cfg.min_det_conf:
            return UpdateSkipReason.LOW_DET_CONF
        if score < self.cfg.update_threshold:
            return UpdateSkipReason.LOW_SIMILARITY
        if margin < self.cfg.margin_threshold:
            return UpdateSkipReason.SMALL_MARGIN
        if not self._track_stable(st):
            return UpdateSkipReason.TRACK_UNSTABLE
        if (st.reacquired_frame is not None) and (st.reacquired_frame + self.cfg.reacquire_cooldown_frames > st.last_seen_frame):
            return UpdateSkipReason.RECENTLY_REACQUIRED

        area, aspect = self._bbox_area_aspect(track)
        if aspect < self.cfg.aspect_ratio_min or aspect > self.cfg.aspect_ratio_max:
            return UpdateSkipReason.SUSPECT_OCCLUSION
        if len(st.area_hist) >= 10:
            med = float(np.median(np.array(st.area_hist, dtype=np.float32)))
            if med > 1e-6 and (area / med) < self.cfg.area_drop_ratio:
                return UpdateSkipReason.SUSPECT_OCCLUSION
        return None

    def assign(self, frame_id: int, track: Track, embedding: Optional[np.ndarray], ts: float) -> Assignment:
        st = self._step_track_state(frame_id, track)
        if embedding is None:
            return Assignment(
                track_id=int(track.track_id),
                identity_id="unknown",
                label="Unknown",
                score=-1.0,
                margin=0.0,
                enrolled=False,
                updated=False,
                skip_reason=UpdateSkipReason.NO_EMBEDDING.value,
            )

        match = self.gallery.match(embedding)
        best_id = match.best_id or "unknown"
        best_score = match.best_score
        margin = match.margin

        enrolled = False
        updated = False
        skip_reason: Optional[UpdateSkipReason] = None

        is_known = match.is_known and best_score >= self.cfg.known_threshold
        if not is_known and best_score < self.cfg.unknown_threshold:
            if self._track_stable(st):
                new_id = self.gallery.add(None, embedding, ts, meta={"track_id": int(track.track_id)})
                best_id = new_id
                best_score = 1.0
                margin = 1.0
                enrolled = True
            else:
                best_id = "unknown"
        elif is_known:
            skip_reason = self._should_update(st, track, best_score, margin)
            if skip_reason is None:
                self.gallery.update(best_id, embedding, ts)
                updated = True
        else:
            skip_reason = UpdateSkipReason.NOT_KNOWN

        label = "Known" if best_id != "unknown" else "Unknown"
        return Assignment(
            track_id=int(track.track_id),
            identity_id=best_id,
            label=label,
            score=float(best_score),
            margin=float(margin),
            enrolled=enrolled,
            updated=updated,
            skip_reason=skip_reason.value if skip_reason else None,
        )
