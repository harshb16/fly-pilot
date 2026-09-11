"""CLI: python -m fly_pilot.brain.query — inspect annotated populations."""

from __future__ import annotations

import argparse
import json
import sys

from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.populations import NeuronIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count MaleCNS neurons matching annotation filters. Requires prepared data."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--superclass")
    parser.add_argument("--class-name")
    parser.add_argument("--subclass")
    parser.add_argument("--type")
    parser.add_argument("--flywire-type")
    parser.add_argument("--cell-type", help="Matches type or flywireType (R1-6 / R1-R6 aliases)")
    parser.add_argument("--side", help="somaSide/rootSide, with instance _L/_R fallback")
    parser.add_argument("--instance-contains")
    parser.add_argument("--consensus-nt")
    parser.add_argument("--inhibitory", action="store_true")
    parser.add_argument("--excitatory", action="store_true")
    parser.add_argument("--list-superclasses", action="store_true")
    parser.add_argument("--list-types", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-ids", action="store_true")
    parser.add_argument("--limit-ids", type=int, default=20)
    args = parser.parse_args(argv)

    connectome = Connectome.load(args.data_dir)
    index = NeuronIndex(connectome)
    payload: dict = {
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "dataset": connectome.manifest.get("dataset"),
    }

    if args.list_superclasses:
        payload["superclasses"] = index.value_counts("superclass")
    if args.list_types:
        payload["types"] = index.value_counts("type", top=args.top)

    filters = {
        "superclass": args.superclass,
        "class_name": args.class_name,
        "subclass": args.subclass,
        "type": args.type,
        "flywire_type": args.flywire_type,
        "cell_type": args.cell_type,
        "side": args.side,
        "instance_contains": args.instance_contains,
        "consensus_nt": args.consensus_nt,
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    if args.inhibitory:
        filters["inhibitory"] = True
    if args.excitatory:
        filters["inhibitory"] = False

    if filters:
        ids = index.query(**filters)
        payload["filters"] = {k: v for k, v in filters.items()}
        payload["match_count"] = int(ids.size)
        if args.show_ids:
            payload["body_ids"] = connectome.body_ids[ids][: args.limit_ids].tolist()
            payload["examples"] = [
                {
                    "body_id": int(connectome.body_ids[i]),
                    "type": str(connectome.type[i]),
                    "flywire_type": str(connectome.flywire_type[i]),
                    "instance": str(connectome.instance[i]),
                    "superclass": str(connectome.superclass[i]),
                    "side": str(connectome.side[i]),
                    "consensus_nt": str(connectome.consensus_nt[i]),
                    "sign": int(connectome.sign[i]),
                }
                for i in ids[: args.limit_ids]
            ]

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"MaleCNS {payload.get('dataset')}  neurons={connectome.n_neurons:,}  edges={connectome.n_edges:,}")
    if "superclasses" in payload:
        print("\nsuperclass counts:")
        for name, count in payload["superclasses"]:
            label = name if name else "<empty>"
            print(f"  {count:7d}  {label}")
    if "types" in payload:
        print(f"\ntop {args.top} types:")
        for name, count in payload["types"]:
            label = name if name else "<empty>"
            print(f"  {count:7d}  {label}")
    if filters:
        print(f"\nfilters: {filters}")
        print(f"matches: {payload['match_count']:,}")
        if args.show_ids:
            for row in payload.get("examples", []):
                print(
                    f"  {row['body_id']:>12}  {row['type'] or row['flywire_type']:<16} "
                    f"{row['superclass']:<22} side={row['side'] or '-':<3} "
                    f"nt={row['consensus_nt']:<14} sign={row['sign']:+d}  {row['instance']}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
