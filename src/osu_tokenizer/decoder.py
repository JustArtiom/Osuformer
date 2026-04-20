from __future__ import annotations

from .events import Event
from .special_tokens import SpecialToken
from .vocab import Vocab


def decode(token_ids: list[int], vocab: Vocab) -> list[Event | SpecialToken]:
    return [vocab.decode_token(tid) for tid in token_ids]
