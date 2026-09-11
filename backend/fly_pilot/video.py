"""Simple approach videos from recorded FLY CONTROL traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
except ImportError:  # pragma: no cover
    plt = None
    animation = None


@dataclass
class TraceFrame:
    sim_time_s: float
    along_m: float
    right_m: float
    alt_agl_m: float
    heading_deg: float
    airspeed_kts: float
    aileron: float
    elevator: float
    rudder: float
    throttle: float
    status: str = "in_progress"


def write_approach_video(frames: list[TraceFrame], path: Path, *, title: str = "FLY CONTROL") -> Path | None:
    if not frames or plt is None or animation is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    along = np.array([f.along_m for f in frames])
    right = np.array([f.right_m for f in frames])
    alt = np.array([f.alt_agl_m for f in frames])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(title + "\nfixed MaleCNS + trained temporal decoder (not biological learning)", fontsize=10)
    ax0, ax1 = axes
    ax0.set_title("top-down")
    ax0.set_xlabel("along (m)")
    ax0.set_ylabel("right (m)")
    ax0.axhspan(-15, 15, xmin=0, xmax=1, color="#3a3a3a", alpha=0.4)
    ax0.plot([0, 1200], [0, 0], color="#f3d48a", lw=1, ls="--")
    ax0.set_xlim(min(-3500, along.min() - 100), max(1400, along.max() + 50))
    ax0.set_ylim(min(-200, right.min() - 20), max(200, right.max() + 20))
    ax0.set_aspect("equal", adjustable="box")
    ax1.set_title("profile")
    ax1.set_xlabel("along (m)")
    ax1.set_ylabel("AGL (m)")
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_ylim(0, max(80, alt.max() + 20))
    (trail0,) = ax0.plot([], [], color="#7ad0a5", lw=2)
    (dot0,) = ax0.plot([], [], "o", color="#e8eef5")
    (trail1,) = ax1.plot([], [], color="#7ad0a5", lw=2)
    (dot1,) = ax1.plot([], [], "o", color="#e8eef5")
    hud = fig.text(0.5, 0.02, "", ha="center", fontsize=8, color="#9fb0c3")

    def _draw(i: int):
        sl = slice(0, i + 1)
        trail0.set_data(along[sl], right[sl])
        dot0.set_data([along[i]], [right[i]])
        trail1.set_data(along[sl], alt[sl])
        dot1.set_data([along[i]], [alt[i]])
        f = frames[i]
        hud.set_text(
            f"t={f.sim_time_s:5.1f}s  IAS {f.airspeed_kts:5.1f}kt  "
            f"ail {f.aileron:+.2f}  elv {f.elevator:+.2f}  rdr {f.rudder:+.2f}  thr {f.throttle:.2f}  {f.status}"
        )
        return trail0, dot0, trail1, dot1, hud

    ani = animation.FuncAnimation(fig, _draw, frames=len(frames), interval=80, blit=False)
    ani.save(path, writer="pillow", fps=12, dpi=90)
    plt.close(fig)
    return path
