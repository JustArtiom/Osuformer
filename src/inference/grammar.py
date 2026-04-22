from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab


class GrammarPhase(Enum):
    BEFORE_OBJECT = "before_object"
    HEADER_FRESH = "header_fresh"
    IN_SLIDER_ANCHORS = "in_slider_anchors"
    NEED_ANCHOR_POS = "need_anchor_pos"
    SLIDER_END_HEADER = "slider_end_header"
    AFTER_LAST_ANCHOR = "after_last_anchor"
    SPINNER_BODY = "spinner_body"
    SPINNER_END_HEADER = "spinner_end_header"


_HEADER_MODIFIERS: tuple[EventType, ...] = (
    EventType.SNAPPING,
    EventType.DISTANCE,
    EventType.POS,
    EventType.HITSOUND,
    EventType.VOLUME,
    EventType.NEW_COMBO,
)
_HEADER_TIMING: tuple[EventType, ...] = (
    EventType.TIMING_POINT,
    EventType.SCROLL_SPEED,
    EventType.KIAI,
    EventType.BEAT,
    EventType.MEASURE,
)
_HEADER_MARKERS: tuple[EventType, ...] = (
    EventType.CIRCLE,
    EventType.SPINNER,
    EventType.SLIDER_HEAD,
)
_ANCHOR_TYPES: tuple[EventType, ...] = (
    EventType.BEZIER_ANCHOR,
    EventType.PERFECT_ANCHOR,
    EventType.CATMULL_ANCHOR,
    EventType.LINEAR_ANCHOR,
    EventType.RED_ANCHOR,
)


class GrammarState:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self._vocab_out = vocab.vocab_size_out
        self._masks: dict[GrammarPhase, Tensor] = self._build_masks()
        self.phase: GrammarPhase = GrammarPhase.BEFORE_OBJECT

    def reset(self) -> None:
        self.phase = GrammarPhase.BEFORE_OBJECT

    def current_mask(self) -> Tensor:
        return self._masks[self.phase]

    def update(self, token_id: int) -> None:
        if token_id == int(SpecialToken.EOS):
            return
        if token_id < 0 or token_id >= self._vocab_out:
            return
        decoded = self.vocab.decode_token(token_id)
        if not isinstance(decoded, Event):
            return
        et = decoded.type
        self.phase = _transition(self.phase, et)

    def _build_masks(self) -> dict[GrammarPhase, Tensor]:
        v = self._vocab_out
        masks: dict[GrammarPhase, Tensor] = {}

        def mk(types: tuple[EventType, ...], specials: tuple[SpecialToken, ...] = ()) -> Tensor:
            m = torch.zeros(v, dtype=torch.bool)
            for t in types:
                start, end = self.vocab.token_range(t)
                m[start:end] = True
            for s in specials:
                sid = int(s)
                if 0 <= sid < v:
                    m[sid] = True
            return m

        masks[GrammarPhase.BEFORE_OBJECT] = mk((EventType.ABS_TIME,), (SpecialToken.EOS,))
        masks[GrammarPhase.HEADER_FRESH] = mk(_HEADER_MODIFIERS + _HEADER_TIMING + _HEADER_MARKERS)
        masks[GrammarPhase.IN_SLIDER_ANCHORS] = mk(_ANCHOR_TYPES + (EventType.POS, EventType.ABS_TIME))
        masks[GrammarPhase.NEED_ANCHOR_POS] = mk((EventType.POS,))
        masks[GrammarPhase.SLIDER_END_HEADER] = mk((EventType.SNAPPING, EventType.DISTANCE, EventType.POS, EventType.LAST_ANCHOR))
        masks[GrammarPhase.AFTER_LAST_ANCHOR] = mk((EventType.SLIDER_SLIDES, EventType.SLIDER_END))
        masks[GrammarPhase.SPINNER_BODY] = mk((EventType.ABS_TIME,))
        masks[GrammarPhase.SPINNER_END_HEADER] = mk((EventType.SNAPPING, EventType.SPINNER_END))
        return masks


def _transition(phase: GrammarPhase, event_type: EventType) -> GrammarPhase:
    if phase == GrammarPhase.BEFORE_OBJECT:
        if event_type == EventType.ABS_TIME:
            return GrammarPhase.HEADER_FRESH
        return phase
    if phase == GrammarPhase.HEADER_FRESH:
        if event_type == EventType.CIRCLE:
            return GrammarPhase.BEFORE_OBJECT
        if event_type == EventType.SPINNER:
            return GrammarPhase.SPINNER_BODY
        if event_type == EventType.SLIDER_HEAD:
            return GrammarPhase.IN_SLIDER_ANCHORS
        return phase
    if phase == GrammarPhase.IN_SLIDER_ANCHORS:
        if event_type in _ANCHOR_TYPES:
            return GrammarPhase.NEED_ANCHOR_POS
        if event_type == EventType.ABS_TIME:
            return GrammarPhase.SLIDER_END_HEADER
        return phase
    if phase == GrammarPhase.NEED_ANCHOR_POS:
        if event_type == EventType.POS:
            return GrammarPhase.IN_SLIDER_ANCHORS
        return phase
    if phase == GrammarPhase.SLIDER_END_HEADER:
        if event_type == EventType.LAST_ANCHOR:
            return GrammarPhase.AFTER_LAST_ANCHOR
        return phase
    if phase == GrammarPhase.AFTER_LAST_ANCHOR:
        if event_type == EventType.SLIDER_END:
            return GrammarPhase.BEFORE_OBJECT
        return phase
    if phase == GrammarPhase.SPINNER_BODY:
        if event_type == EventType.ABS_TIME:
            return GrammarPhase.SPINNER_END_HEADER
        return phase
    if phase == GrammarPhase.SPINNER_END_HEADER:
        if event_type == EventType.SPINNER_END:
            return GrammarPhase.BEFORE_OBJECT
        return phase
    return phase
