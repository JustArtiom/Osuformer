from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab


class GrammarPhase(Enum):
    BEFORE_OBJECT = "before_object"
    AFTER_MARKER = "after_marker"
    CIRCLE_HEADER = "circle_header"
    SLIDER_HEADER = "slider_header"
    SPINNER_HEADER = "spinner_header"
    IN_SLIDER_ANCHORS = "in_slider_anchors"
    NEED_ANCHOR_POS = "need_anchor_pos"
    AFTER_SLIDER_DURATION = "after_slider_duration"
    SPINNER_AFTER_DURATION = "spinner_after_duration"


_HEADER_MODIFIERS: tuple[EventType, ...] = (
    EventType.SNAPPING,
    EventType.DISTANCE,
    EventType.POS,
    EventType.HITSOUND,
    EventType.VOLUME,
    EventType.NEW_COMBO,
)
_TIMING_EVENTS: tuple[EventType, ...] = (
    EventType.TIMING_POINT,
    EventType.SCROLL_SPEED,
    EventType.KIAI,
    EventType.BEAT,
    EventType.MEASURE,
)
_OBJECT_MARKERS: tuple[EventType, ...] = (
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
        self._pending_marker: EventType | None = None

    def reset(self) -> None:
        self.phase = GrammarPhase.BEFORE_OBJECT
        self._pending_marker = None

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
        if self.phase == GrammarPhase.BEFORE_OBJECT and et in _OBJECT_MARKERS:
            self._pending_marker = et
            self.phase = GrammarPhase.AFTER_MARKER
            return
        if self.phase == GrammarPhase.AFTER_MARKER and et == EventType.ABS_TIME:
            if self._pending_marker == EventType.CIRCLE:
                self.phase = GrammarPhase.CIRCLE_HEADER
            elif self._pending_marker == EventType.SLIDER_HEAD:
                self.phase = GrammarPhase.SLIDER_HEADER
            elif self._pending_marker == EventType.SPINNER:
                self.phase = GrammarPhase.SPINNER_HEADER
            self._pending_marker = None
            return
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

        masks[GrammarPhase.BEFORE_OBJECT] = mk(
            _OBJECT_MARKERS + (EventType.ABS_TIME,), (SpecialToken.EOS,)
        )
        masks[GrammarPhase.AFTER_MARKER] = mk((EventType.ABS_TIME,))
        masks[GrammarPhase.CIRCLE_HEADER] = mk(
            _HEADER_MODIFIERS + _TIMING_EVENTS + _OBJECT_MARKERS + (EventType.ABS_TIME,),
            (SpecialToken.EOS,),
        )
        masks[GrammarPhase.SLIDER_HEADER] = mk(_HEADER_MODIFIERS + _ANCHOR_TYPES)
        masks[GrammarPhase.SPINNER_HEADER] = mk(_HEADER_MODIFIERS + (EventType.DURATION,))
        masks[GrammarPhase.IN_SLIDER_ANCHORS] = mk(_ANCHOR_TYPES + (EventType.DURATION,))
        masks[GrammarPhase.NEED_ANCHOR_POS] = mk((EventType.POS,))
        masks[GrammarPhase.AFTER_SLIDER_DURATION] = mk((EventType.SLIDER_SLIDES, EventType.SLIDER_END))
        masks[GrammarPhase.SPINNER_AFTER_DURATION] = mk((EventType.SPINNER_END,))
        return masks


def _transition(phase: GrammarPhase, event_type: EventType) -> GrammarPhase:
    if phase == GrammarPhase.CIRCLE_HEADER:
        if event_type in _OBJECT_MARKERS:
            return GrammarPhase.AFTER_MARKER
        if event_type == EventType.ABS_TIME:
            return GrammarPhase.CIRCLE_HEADER
        return phase
    if phase == GrammarPhase.SLIDER_HEADER:
        if event_type in _ANCHOR_TYPES:
            return GrammarPhase.NEED_ANCHOR_POS
        return phase
    if phase == GrammarPhase.SPINNER_HEADER:
        if event_type == EventType.DURATION:
            return GrammarPhase.SPINNER_AFTER_DURATION
        return phase
    if phase == GrammarPhase.IN_SLIDER_ANCHORS:
        if event_type in _ANCHOR_TYPES:
            return GrammarPhase.NEED_ANCHOR_POS
        if event_type == EventType.DURATION:
            return GrammarPhase.AFTER_SLIDER_DURATION
        return phase
    if phase == GrammarPhase.NEED_ANCHOR_POS:
        if event_type == EventType.POS:
            return GrammarPhase.IN_SLIDER_ANCHORS
        return phase
    if phase == GrammarPhase.AFTER_SLIDER_DURATION:
        if event_type == EventType.SLIDER_END:
            return GrammarPhase.BEFORE_OBJECT
        return phase
    if phase == GrammarPhase.SPINNER_AFTER_DURATION:
        if event_type == EventType.SPINNER_END:
            return GrammarPhase.BEFORE_OBJECT
        return phase
    return phase
