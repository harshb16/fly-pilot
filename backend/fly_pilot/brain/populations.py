"""Query neurons by MaleCNS annotations (type, superclass, side, …)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_pilot.brain.connectome import Connectome


def _norm(value: str) -> str:
    return str(value).strip().lower()


def _match_field(column: np.ndarray, wanted: str) -> np.ndarray:
    target = _norm(wanted)
    return np.fromiter((_norm(v) == target for v in column), dtype=bool, count=len(column))


def _match_cell_type(connectome: Connectome, wanted: str) -> np.ndarray:
    target = _norm(wanted)
    aliases = {target}
    compact = target.replace("-", "").replace("_", "")
    aliases.add(compact)
    if target == "r1-6":
        aliases.add("r1-r6")
    if target == "r1-r6":
        aliases.add("r1-6")
    out = np.zeros(connectome.n_neurons, dtype=bool)
    for column in (connectome.type, connectome.flywire_type):
        for i, value in enumerate(column):
            text = _norm(value)
            if text in aliases or text.replace("-", "").replace("_", "") == compact:
                out[i] = True
    return out


@dataclass
class NeuronIndex:
    """Annotation lookup over a loaded connectome. Not limited to DNs."""

    connectome: Connectome

    def query(
        self,
        *,
        superclass: str | None = None,
        class_name: str | None = None,
        subclass: str | None = None,
        type: str | None = None,
        flywire_type: str | None = None,
        cell_type: str | None = None,
        side: str | None = None,
        instance_contains: str | None = None,
        consensus_nt: str | None = None,
        inhibitory: bool | None = None,
    ) -> np.ndarray:
        """Return dense neuron indices matching all provided filters."""
        mask = np.ones(self.connectome.n_neurons, dtype=bool)
        if superclass is not None:
            mask &= _match_field(self.connectome.superclass, superclass)
        if class_name is not None:
            mask &= _match_field(self.connectome.class_name, class_name)
        if subclass is not None:
            mask &= _match_field(self.connectome.subclass, subclass)
        if type is not None:
            mask &= _match_field(self.connectome.type, type)
        if flywire_type is not None:
            mask &= _match_field(self.connectome.flywire_type, flywire_type)
        if cell_type is not None:
            mask &= _match_cell_type(self.connectome, cell_type)
        if side is not None:
            wanted = _norm(side)
            side_hit = np.fromiter((_norm(v) == wanted for v in self.connectome.side), dtype=bool, count=self.connectome.n_neurons)
            inst_hit = np.fromiter(
                (f"_{wanted}" in _norm(v) or _norm(v).endswith(f"({wanted})") for v in self.connectome.instance),
                dtype=bool,
                count=self.connectome.n_neurons,
            )
            mask &= side_hit | inst_hit
        if instance_contains is not None:
            needle = _norm(instance_contains)
            mask &= np.fromiter((needle in _norm(v) for v in self.connectome.instance), dtype=bool, count=self.connectome.n_neurons)
        if consensus_nt is not None:
            mask &= _match_field(self.connectome.consensus_nt, consensus_nt)
        if inhibitory is True:
            mask &= self.connectome.sign < 0
        elif inhibitory is False:
            mask &= self.connectome.sign > 0
        return np.flatnonzero(mask).astype(np.int32)

    def count(self, **filters) -> int:
        return int(self.query(**filters).size)

    def body_ids(self, **filters) -> np.ndarray:
        return self.connectome.body_ids[self.query(**filters)]

    def value_counts(self, field: str, *, top: int | None = None) -> list[tuple[str, int]]:
        column = {
            "superclass": self.connectome.superclass,
            "class": self.connectome.class_name,
            "subclass": self.connectome.subclass,
            "type": self.connectome.type,
            "flywire_type": self.connectome.flywire_type,
            "side": self.connectome.side,
            "consensus_nt": self.connectome.consensus_nt,
        }[field]
        values, counts = np.unique(column.astype(str), return_counts=True)
        order = np.argsort(-counts)
        pairs = [(str(values[i]), int(counts[i])) for i in order]
        if top is not None:
            return pairs[:top]
        return pairs
