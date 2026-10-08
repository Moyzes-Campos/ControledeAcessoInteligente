from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import time
from uuid import uuid4

from .access_store import AccessStore
from .line_crossing import LineCounter
from .tracking import bbox_center
from .types import TrackIdentity, TrackedPose


def signal_for_tracks(tracks: list[TrackedPose], identities: dict[int, TrackIdentity]) -> str:
    if not tracks:
        return "orange"
    return "green" if all(identities[track.track_id].known for track in tracks) else "red"


@dataclass(slots=True)
class AccessState:
    lamp: str
    alarm_until: float
    entries: int
    exits: int
    open_visits: int
    events: list[dict]


class AccessController:
    def __init__(self, counter: LineCounter, store: AccessStore, alarm_seconds: float = 3.0):
        if not math.isfinite(alarm_seconds) or alarm_seconds <= 0:
            raise ValueError("A duracao do alerta deve ser positiva e finita")
        self.counter = counter
        self.store = store
        self.alarm_seconds = alarm_seconds
        self.run_id = uuid4().hex
        self.alarm_until = 0.0
        self.unknown_crossings = 0
        self._pending: dict[int, list[int]] = {}

    def update(self, tracks: list[TrackedPose], identities: dict[int, TrackIdentity],
               active_track_ids: set[int], timestamp: datetime,
               monotonic_time: float | None = None) -> AccessState:
        clock = time.monotonic() if monotonic_time is None else monotonic_time
        events = []
        # A face may become visible after the crossing. Preserve the crossing time.
        for track in tracks:
            identity = identities[track.track_id]
            if identity.known:
                for event_id in self._pending.pop(track.track_id, []):
                    events.append(self.store.identify_crossing(event_id, identity.label))
            point = bbox_center(track.bbox_xyxy)
            crossing = self.counter.update(track.track_id, point)
            if crossing is None:
                continue
            event = self.store.register_crossing(
                self.run_id, track.track_id, crossing.direction, timestamp,
                identity.label if identity.known else None,
            )
            events.append(event)
            if not identity.known:
                self._pending.setdefault(track.track_id, []).append(event["id"])
                self.unknown_crossings += 1
                self.alarm_until = max(self.alarm_until, clock + self.alarm_seconds)

        self.counter.prune(active_track_ids)
        for track_id in list(self._pending):
            if track_id not in active_track_ids:
                del self._pending[track_id]

        lamp = signal_for_tracks(tracks, identities)
        return AccessState(lamp, self.alarm_until, self.counter.entries, self.counter.exits,
                           self.store.open_count, events)
