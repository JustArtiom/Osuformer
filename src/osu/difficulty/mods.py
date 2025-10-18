from __future__ import annotations

from typing import Iterable, List, Sequence

from ..mods import Mods


MOD_CLOCK_RATES = {
    "DoubleTime": 1.5,
    "NightCore": 1.5,
    "HalfTime": 0.75,
}


def normalise_mods(mods: Sequence[Mods | str] | None) -> List[str]:
    if not mods:
        return []

    result: List[str] = []
    for mod in mods:
        if isinstance(mod, Mods):
            result.append(mod.name)
        elif isinstance(mod, str):
            result.append(mod.strip())
        else:
            raise TypeError(f"Unsupported mod type: {type(mod)!r}")
    return result


def clock_rate_for_mods(mods: Iterable[str]) -> float:
    rate = 1.0
    for mod in mods:
        rate *= MOD_CLOCK_RATES.get(mod, 1.0)
    return rate

