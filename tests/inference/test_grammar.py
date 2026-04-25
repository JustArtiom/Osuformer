from src.config.loader import load_config
from src.inference.grammar import GrammarPhase, GrammarState
from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab


def _vocab() -> Vocab:
    return Vocab(load_config("config/config.yaml").tokenizer)


def _encode(vocab: Vocab, event_type: EventType, value: int = 0) -> int:
    return vocab.encode_event(Event(type=event_type, value=value))


def test_before_object_allows_markers_and_abs_time_and_eos() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    mask = g.current_mask()
    for marker in (EventType.CIRCLE, EventType.SLIDER_HEAD, EventType.SPINNER):
        start, end = vocab.token_range(marker)
        assert mask[start:end].all()
    abs_start, abs_end = vocab.token_range(EventType.ABS_TIME)
    assert mask[abs_start:abs_end].all()
    assert mask[int(SpecialToken.EOS)]


def test_circle_flow() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.CIRCLE, 0))
    assert g.phase == GrammarPhase.AFTER_MARKER
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    assert g.phase == GrammarPhase.CIRCLE_HEADER
    for t in (EventType.SNAPPING, EventType.DISTANCE, EventType.POS, EventType.HITSOUND, EventType.VOLUME):
        g.update(_encode(vocab, t, 0))
        assert g.phase == GrammarPhase.CIRCLE_HEADER
    g.update(_encode(vocab, EventType.CIRCLE, 0))
    assert g.phase == GrammarPhase.AFTER_MARKER


def test_slider_full_flow() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    assert g.phase == GrammarPhase.AFTER_MARKER
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    assert g.phase == GrammarPhase.SLIDER_HEADER
    for t in (EventType.SNAPPING, EventType.DISTANCE, EventType.POS, EventType.HITSOUND, EventType.VOLUME):
        g.update(_encode(vocab, t, 0))
        assert g.phase == GrammarPhase.SLIDER_HEADER
    g.update(_encode(vocab, EventType.BEZIER_ANCHOR, 0))
    assert g.phase == GrammarPhase.NEED_ANCHOR_POS
    g.update(_encode(vocab, EventType.POS, 20))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    g.update(_encode(vocab, EventType.LINEAR_ANCHOR, 0))
    assert g.phase == GrammarPhase.NEED_ANCHOR_POS
    g.update(_encode(vocab, EventType.POS, 30))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    g.update(_encode(vocab, EventType.DURATION, 5))
    assert g.phase == GrammarPhase.AFTER_SLIDER_DURATION
    g.update(_encode(vocab, EventType.SLIDER_END, 0))
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def test_spinner_flow() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.SPINNER, 0))
    assert g.phase == GrammarPhase.AFTER_MARKER
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    assert g.phase == GrammarPhase.SPINNER_HEADER
    for t in (EventType.SNAPPING, EventType.POS, EventType.VOLUME):
        g.update(_encode(vocab, t, 0))
        assert g.phase == GrammarPhase.SPINNER_HEADER
    g.update(_encode(vocab, EventType.DURATION, 10))
    assert g.phase == GrammarPhase.SPINNER_AFTER_DURATION
    g.update(_encode(vocab, EventType.SPINNER_END, 0))
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def test_mask_in_slider_anchors_allows_anchors_and_duration() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.POS, 0))
    g.update(_encode(vocab, EventType.BEZIER_ANCHOR, 0))
    g.update(_encode(vocab, EventType.POS, 10))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    mask = g.current_mask()
    bez_start, bez_end = vocab.token_range(EventType.BEZIER_ANCHOR)
    dur_start, dur_end = vocab.token_range(EventType.DURATION)
    assert mask[bez_start:bez_end].all()
    assert mask[dur_start:dur_end].all()
    circle_start, circle_end = vocab.token_range(EventType.CIRCLE)
    assert not mask[circle_start:circle_end].any()


def test_need_anchor_pos_only_allows_pos() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.BEZIER_ANCHOR, 0))
    assert g.phase == GrammarPhase.NEED_ANCHOR_POS
    mask = g.current_mask()
    pos_start, pos_end = vocab.token_range(EventType.POS)
    assert mask[pos_start:pos_end].all()
    bez_start, bez_end = vocab.token_range(EventType.BEZIER_ANCHOR)
    assert not mask[bez_start:bez_end].any()


def test_after_duration_allows_slides_and_end_only() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.phase = GrammarPhase.AFTER_SLIDER_DURATION
    mask = g.current_mask()
    slides_start, slides_end = vocab.token_range(EventType.SLIDER_SLIDES)
    end_start, end_end = vocab.token_range(EventType.SLIDER_END)
    assert mask[slides_start:slides_end].all()
    assert mask[end_start:end_end].all()
    abs_start, abs_end = vocab.token_range(EventType.ABS_TIME)
    assert not mask[abs_start:abs_end].any()


def test_reset_returns_to_before_object() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    assert g.phase == GrammarPhase.SLIDER_HEADER
    g.reset()
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def _drive(vocab: Vocab, types: list[tuple[EventType, int]]) -> GrammarState:
    g = GrammarState(vocab)
    for t, v in types:
        token_id = _encode(vocab, t, v)
        mask = g.current_mask()
        if token_id < mask.numel():
            assert mask[token_id], f"token {t.name}({v}) not allowed in phase {g.phase}"
        g.update(token_id)
    return g


def test_two_circles_back_to_back_realistic_sequence() -> None:
    vocab = _vocab()
    seq: list[tuple[EventType, int]] = [
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 100),
        (EventType.SNAPPING, 4),
        (EventType.DISTANCE, 50),
        (EventType.POS, 100),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 200),
        (EventType.SNAPPING, 4),
        (EventType.DISTANCE, 30),
        (EventType.POS, 200),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
    ]
    g = _drive(vocab, seq)
    assert g.phase == GrammarPhase.CIRCLE_HEADER


def test_circle_then_slider_then_circle_sequence() -> None:
    vocab = _vocab()
    seq: list[tuple[EventType, int]] = [
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 100),
        (EventType.SNAPPING, 4),
        (EventType.DISTANCE, 0),
        (EventType.POS, 50),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
        (EventType.SLIDER_HEAD, 0),
        (EventType.ABS_TIME, 200),
        (EventType.SNAPPING, 4),
        (EventType.DISTANCE, 50),
        (EventType.POS, 100),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
        (EventType.BEZIER_ANCHOR, 0),
        (EventType.POS, 150),
        (EventType.DURATION, 5),
        (EventType.SLIDER_END, 0),
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 400),
        (EventType.SNAPPING, 4),
        (EventType.POS, 200),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
    ]
    g = _drive(vocab, seq)
    assert g.phase == GrammarPhase.CIRCLE_HEADER


def test_circle_then_timing_group_then_circle() -> None:
    vocab = _vocab()
    seq: list[tuple[EventType, int]] = [
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 100),
        (EventType.SNAPPING, 4),
        (EventType.POS, 50),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
        (EventType.ABS_TIME, 150),
        (EventType.BEAT, 0),
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 200),
        (EventType.SNAPPING, 4),
        (EventType.POS, 100),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
    ]
    g = _drive(vocab, seq)
    assert g.phase == GrammarPhase.CIRCLE_HEADER


def test_timing_point_followed_by_kiai_inside_timing_group() -> None:
    vocab = _vocab()
    seq: list[tuple[EventType, int]] = [
        (EventType.ABS_TIME, 50),
        (EventType.TIMING_POINT, 0),
        (EventType.KIAI, 1),
        (EventType.CIRCLE, 0),
        (EventType.ABS_TIME, 100),
        (EventType.SNAPPING, 4),
        (EventType.POS, 50),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
    ]
    g = _drive(vocab, seq)
    assert g.phase == GrammarPhase.CIRCLE_HEADER


def test_scroll_speed_then_kiai_then_marker() -> None:
    vocab = _vocab()
    seq: list[tuple[EventType, int]] = [
        (EventType.ABS_TIME, 50),
        (EventType.SCROLL_SPEED, 100),
        (EventType.KIAI, 0),
        (EventType.SLIDER_HEAD, 0),
        (EventType.ABS_TIME, 100),
        (EventType.SNAPPING, 4),
        (EventType.POS, 50),
        (EventType.HITSOUND, 0),
        (EventType.VOLUME, 80),
        (EventType.LINEAR_ANCHOR, 0),
        (EventType.POS, 100),
        (EventType.DURATION, 8),
        (EventType.SLIDER_END, 0),
    ]
    g = _drive(vocab, seq)
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def test_long_realistic_sequence_does_not_get_stuck() -> None:
    vocab = _vocab()
    seq: list[tuple[EventType, int]] = []
    for i in range(20):
        seq += [
            (EventType.CIRCLE, 0),
            (EventType.ABS_TIME, 100 + i * 10),
            (EventType.SNAPPING, 4),
            (EventType.POS, 50 + i),
            (EventType.HITSOUND, 0),
            (EventType.VOLUME, 80),
        ]
    g = _drive(vocab, seq)
    assert g.phase == GrammarPhase.CIRCLE_HEADER
