
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from utils import MovingAverage


class SquatPhase(Enum):
    STANDING   = "Standing"
    DESCENDING = "Descending"
    BOTTOM     = "At Bottom"
    ASCENDING  = "Ascending"
    CALIBRATING = "Calibrating…"


@dataclass
class RepRecord:
    rep_number: int
    score: float
    issues: List[str] = field(default_factory=list)
    depth_ok: bool = True


class RepCounter:

    def __init__(
        self,
        descent_threshold: float = 0.04,
        bottom_threshold: float = 0.15,
        ascent_threshold: float = 0.05,
        complete_threshold: float = 0.06,
        smoothing_window: int = 7,
    ):
        self._descent_thresh = descent_threshold
        self._bottom_thresh = bottom_threshold
        self._ascent_thresh = ascent_threshold
        self._complete_thresh = complete_threshold

        self._smoother = MovingAverage(window=smoothing_window)

        self.phase: SquatPhase = SquatPhase.CALIBRATING
        self.rep_count: int = 0
        self.rep_history: List[RepRecord] = []

        self._calibrated: bool = False
        self._calib_smoother = MovingAverage(window=30)
        self._standing_hip_y: Optional[float] = None
        self._bottom_hip_y: Optional[float] = None

        self._on_rep_complete: Optional[Callable[[RepRecord], None]] = None


    def on_rep_complete(self, callback: Callable[[RepRecord], None]):
        self._on_rep_complete = callback

    def update(self, hip_y_pixel: float, frame_height: int) -> SquatPhase:
        smoothed_y = self._smoother.update(hip_y_pixel)

        if not self._calibrated:
            ref_y = self._calib_smoother.update(smoothed_y)
            self._standing_hip_y = ref_y
            if len(self._calib_smoother._buf) >= 28:
                self._calibrated = True
                self._standing_hip_y = ref_y
                self.phase = SquatPhase.STANDING
            else:
                self.phase = SquatPhase.CALIBRATING
            return self.phase

        ref = self._standing_hip_y
        H = float(frame_height)
        descent = (smoothed_y - ref) / H

        if self.phase == SquatPhase.STANDING:
            if descent > self._descent_thresh:
                self.phase = SquatPhase.DESCENDING
                self._bottom_hip_y = smoothed_y

        elif self.phase == SquatPhase.DESCENDING:
            if smoothed_y > (self._bottom_hip_y or smoothed_y):
                self._bottom_hip_y = smoothed_y

            if descent > self._bottom_thresh:
                self.phase = SquatPhase.BOTTOM
            elif descent < self._descent_thresh * 0.5:
                self.phase = SquatPhase.STANDING

        elif self.phase == SquatPhase.BOTTOM:
            if smoothed_y > (self._bottom_hip_y or smoothed_y):
                self._bottom_hip_y = smoothed_y

            bottom_ref = self._bottom_hip_y or smoothed_y
            rise = (bottom_ref - smoothed_y) / H
            if rise > self._ascent_thresh:
                self.phase = SquatPhase.ASCENDING

        elif self.phase == SquatPhase.ASCENDING:
            if descent < self._complete_thresh:
                self._complete_rep()
                self.phase = SquatPhase.STANDING

        return self.phase

    def record_rep_score(self, score: float, issues: List[str], depth_ok: bool):
        if self.rep_history:
            last = self.rep_history[-1]
            last.score = score
            last.issues = issues
            last.depth_ok = depth_ok

    def reset(self):
        self._smoother.reset()
        self._calib_smoother.reset()
        self.phase = SquatPhase.CALIBRATING
        self.rep_count = 0
        self.rep_history.clear()
        self._calibrated = False
        self._standing_hip_y = None
        self._bottom_hip_y = None

    def recalibrate(self):
        self._smoother.reset()
        self._calib_smoother.reset()
        self._calibrated = False
        self._standing_hip_y = None
        self._bottom_hip_y = None
        self.phase = SquatPhase.CALIBRATING

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def average_score(self) -> float:
        scores = [r.score for r in self.rep_history if r.score > 0]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @property
    def most_frequent_issue(self) -> Optional[str]:
        from collections import Counter
        all_issues = [i for r in self.rep_history for i in r.issues]
        if not all_issues:
            return None
        return Counter(all_issues).most_common(1)[0][0]


    def _complete_rep(self):
        self.rep_count += 1
        record = RepRecord(
            rep_number=self.rep_count,
            score=0.0,
        )
        self.rep_history.append(record)
        self._bottom_hip_y = None

        if self._on_rep_complete:
            self._on_rep_complete(record)
