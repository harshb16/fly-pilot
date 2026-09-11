"""python -m fly_pilot.brain — list standalone brain commands."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "FlyPilot MaleCNS module — measured connectome, modeled LIF dynamics.\n"
        "Not connected to the aircraft. No retina. No decoder.\n\n"
        "  python -m fly_pilot.brain.prepare     # download/validate/compact MaleCNS v1.0\n"
        "  python -m fly_pilot.brain.query       # inspect annotated populations\n"
        "  python -m fly_pilot.brain.demo        # baseline → stim → recovery\n"
        "  python -m fly_pilot.brain.benchmark   # full-network timing on this machine\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
