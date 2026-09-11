"""Explicit simulation-time scheduler.

Browser frame rate must never determine MaleCNS time. JSBSim, vision, and
neural updates are due when *simulated* time crosses their periods.

Default rates (Milestone 4):

- JSBSim physics: 120 Hz (dt = 1/120 s, owned by the FDM)
- Visual observation / retinal encode: 50 Hz
- MaleCNS LIF: 50 Hz (dt = 20 ms)
- Neural feature logging: 50 Hz (same instants as neural steps)
- Browser state broadcast: 30 Hz (display only)

A recorded retinal sequence replayed at these neural instants must produce the
same spike checksums regardless of how fast the frontend rendered.
"""

from __future__ import annotations

from dataclasses import dataclass


PHYSICS_HZ = 120.0
VISION_HZ = 50.0
NEURAL_HZ = 50.0
LOG_HZ = 50.0
STATE_BROADCAST_HZ = 30.0


@dataclass(frozen=True)
class Schedule:
    """Named rates. Neural and vision share 50 Hz so each LIF step has one frame."""

    physics_hz: float = PHYSICS_HZ
    vision_hz: float = VISION_HZ
    neural_hz: float = NEURAL_HZ
    log_hz: float = LOG_HZ
    state_broadcast_hz: float = STATE_BROADCAST_HZ

    def __post_init__(self) -> None:
        for name in ("physics_hz", "vision_hz", "neural_hz", "log_hz", "state_broadcast_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def vision_dt(self) -> float:
        return 1.0 / self.vision_hz

    @property
    def neural_dt(self) -> float:
        return 1.0 / self.neural_hz

    @property
    def log_dt(self) -> float:
        return 1.0 / self.log_hz

    @property
    def state_broadcast_dt(self) -> float:
        return 1.0 / self.state_broadcast_hz

    def as_dict(self) -> dict[str, float]:
        return {
            "physics_hz": self.physics_hz,
            "vision_hz": self.vision_hz,
            "neural_hz": self.neural_hz,
            "log_hz": self.log_hz,
            "state_broadcast_hz": self.state_broadcast_hz,
            "physics_dt": self.physics_dt,
            "vision_dt": self.vision_dt,
            "neural_dt": self.neural_dt,
        }


@dataclass(frozen=True)
class SchedulerEvents:
    """Which clocks fire after the physics integrator has reached ``sim_time_s``."""

    sim_time_s: float
    physics_steps: int
    vision: bool
    neural: bool
    log: bool
    neural_step: int | None
    vision_step: int | None


class SimScheduler:
    """Advance due flags from simulated time, never from wall or rAF time.

    Neural step ``k`` is due the first time ``sim_time_s >= k * neural_dt``,
    including ``k = 0`` at reset (``sim_time_s = 0``).
    """

    def __init__(self, schedule: Schedule | None = None) -> None:
        self.schedule = schedule or Schedule()
        self.sim_time_s = 0.0
        self.physics_steps = 0
        self.neural_step = 0
        self.vision_step = 0
        self.log_step = 0

    def reset(self) -> None:
        self.sim_time_s = 0.0
        self.physics_steps = 0
        self.neural_step = 0
        self.vision_step = 0
        self.log_step = 0

    def due_at(self, sim_time_s: float) -> SchedulerEvents:
        """Return clocks that should fire at this simulated time, without mutating."""
        sched = self.schedule
        vision = self.vision_step * sched.vision_dt <= sim_time_s + 1e-12
        neural = self.neural_step * sched.neural_dt <= sim_time_s + 1e-12
        log = self.log_step * sched.log_dt <= sim_time_s + 1e-12
        return SchedulerEvents(
            sim_time_s=sim_time_s,
            physics_steps=self.physics_steps,
            vision=vision,
            neural=neural,
            log=log,
            neural_step=self.neural_step if neural else None,
            vision_step=self.vision_step if vision else None,
        )

    def note_physics(self, physics_dt: float) -> float:
        """Record one FDM step. Returns the new simulated time."""
        self.sim_time_s += float(physics_dt)
        self.physics_steps += 1
        return self.sim_time_s

    def consume(self, sim_time_s: float | None = None) -> SchedulerEvents:
        """Mark due clocks as consumed at ``sim_time_s`` (default: current).

        Call this *after* performing the vision/neural/log work indicated by
        :meth:`due_at`. Multiple neural steps are not collapsed: if sim time
        jumps by more than one neural dt, call this in a loop.
        """
        t = self.sim_time_s if sim_time_s is None else float(sim_time_s)
        events = self.due_at(t)
        if events.vision:
            self.vision_step += 1
        if events.neural:
            self.neural_step += 1
        if events.log:
            self.log_step += 1
        return events

    def drain(self, sim_time_s: float | None = None) -> list[SchedulerEvents]:
        """Yield every due neural/vision instant up to ``sim_time_s``.

        Used when several neural ticks fit in one physics step (should be rare
        at 120 Hz physics / 50 Hz neural) or at reset (``t = 0``).
        """
        t = self.sim_time_s if sim_time_s is None else float(sim_time_s)
        out: list[SchedulerEvents] = []
        while True:
            events = self.due_at(t)
            if not (events.vision or events.neural or events.log):
                break
            self.consume(t)
            out.append(events)
        return out
