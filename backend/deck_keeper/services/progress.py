"""Live progress of the scan being processed, for the scan page.

The worker runs in the app's own process, so progress lives in memory: the code doing the work
calls stage() and tick() (a no-op outside a scan) and the scan page polls snapshot() through
the API. Each finished stage keeps its duration, so the page (and the raw data) show what took
the time.
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field

_live: dict[int, "Progress"] = {}
_current: ContextVar["Progress | None"] = ContextVar("progress", default=None)


@dataclass
class Stage:
    name: str
    total: int | None = None
    done: int = 0
    started: float = field(default_factory=time.monotonic)
    seconds: float | None = None


TAIL = 10.0  # seconds the stages after the current one take at least (identifying, the decklist)


@dataclass
class Progress:
    job_id: int
    photo: int = 0
    photos: int = 0
    stage: Stage | None = None
    finished: list[tuple[int, Stage]] = field(default_factory=list)  # (photo, stage)
    estimate: float | None = None  # seconds the whole scan should take (estimate.py)
    started: float = field(default_factory=time.monotonic)

    def remaining(self, now: float) -> float | None:
        """Seconds left: the estimate less the time taken, but never less than the current stage
        needs at the rate it is going (so a slow scan's time left stays honest)."""
        elapsed = now - self.started
        s = self.stage
        stage_left = 0.0
        if s and s.total and s.done:
            stage_left = (s.total - s.done) * (now - s.started) / s.done
        if self.estimate is None:
            return stage_left + TAIL if stage_left else None
        later_photos = max(0, self.photos - self.photo) * self.estimate / max(1, self.photos)
        return max(self.estimate - elapsed, stage_left + later_photos + TAIL)

    def snapshot(self) -> dict:
        now = time.monotonic()
        s = self.stage
        left = self.remaining(now)
        elapsed = now - self.started
        return {
            "photo": self.photo, "photos": self.photos,
            "elapsed": round(elapsed), "remaining": None if left is None else round(left),
            "estimate": None if self.estimate is None else round(self.estimate),
            "fraction": None if left is None else round(elapsed / max(1.0, elapsed + left), 3),
            "stage": s and {"name": s.name, "done": s.done, "total": s.total,
                            "seconds": round(now - s.started, 1)},
            "finished": [{"photo": p, "name": st.name, "seconds": round(st.seconds or 0, 1),
                          "count": st.total} for p, st in self.finished[-12:]],
        }

    def timings(self, photo: int) -> list[dict]:
        return [{"stage": st.name, "seconds": round(st.seconds or 0, 1), "count": st.total}
                for p, st in self.finished if p == photo]


def start(job_id: int, photos: int, estimate: float | None = None) -> Progress:
    p = Progress(job_id, photos=photos, estimate=estimate)
    _live[job_id] = p
    _current.set(p)
    return p


def end(job_id: int) -> None:
    _live.pop(job_id, None)
    _current.set(None)


def get(job_id: int) -> dict | None:
    p = _live.get(job_id)
    return p.snapshot() if p else None


def current() -> Progress | None:
    return _current.get()


def photo(n: int) -> None:
    p = _current.get()
    if p:
        _finish(p)
        p.photo = n


def stage(name: str, total: int | None = None) -> None:
    """Start a stage of work (finishing the one before), with a count of items if known."""
    p = _current.get()
    if p:
        _finish(p)
        p.stage = Stage(name, total)


def finish() -> None:
    """The current stage is done (without starting another)."""
    p = _current.get()
    if p:
        _finish(p)


def tick(n: int = 1) -> None:
    p = _current.get()
    if p and p.stage:
        p.stage.done += n


def add(n: int) -> None:
    """More items turned up in the current stage (e.g. a close-up that needs a second try)."""
    p = _current.get()
    if p and p.stage and p.stage.total is not None:
        p.stage.total += n


def _finish(p: Progress) -> None:
    if p.stage:
        p.stage.seconds = time.monotonic() - p.stage.started
        p.finished.append((p.photo, p.stage))
        p.stage = None
