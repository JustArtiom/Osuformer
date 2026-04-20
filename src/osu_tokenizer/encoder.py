from __future__ import annotations

from .events import Event
from .special_tokens import SpecialToken
from .vocab import Vocab


def encode(tokens: list[Event | SpecialToken], vocab: Vocab) -> list[int]:
    out: list[int] = []
    for token in tokens:
        if isinstance(token, SpecialToken):
            out.append(int(token))
        else:
            out.append(vocab.encode_event(token))
    return out
