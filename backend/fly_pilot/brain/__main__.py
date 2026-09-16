"""python -m fly_pilot.brain — list standalone brain commands."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "FlyPilot MaleCNS module — measured connectome, modeled LIF dynamics.\n"
        "Milestone 5: FLY CONTROL uses a trained temporal decoder on DN activity.\n"
        "MaleCNS synapses are not trained. This is not biological learning.\n\n"
        "  python -m fly_pilot.brain.prepare          # download/validate/compact MaleCNS v1.0\n"
        "  python -m fly_pilot.brain.query            # inspect annotated populations\n"
        "  python -m fly_pilot.brain.demo             # baseline → stim → recovery\n"
        "  python -m fly_pilot.brain.benchmark        # multi-regime timing on this machine\n"
        "  python -m fly_pilot.brain.replay_episode   # replay recorded retinal stimuli\n"
        "  python -m fly_pilot.record_observing       # expert landings + neural features\n"
        "  python -m fly_pilot.record_decoder         # compact DN-rate training set\n"
        "  python -m fly_pilot.train_decoder          # causal GRU decoder\n"
        "  python -m fly_pilot.evaluate_fly           # closed-loop FLY CONTROL\n"
        "  python -m fly_pilot.validate_vision        # experiments A–D\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
