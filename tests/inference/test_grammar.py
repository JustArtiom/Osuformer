from src.config.loader import load_config
from src.inference.grammar import GrammarPhase, GrammarState
from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab


def _vocab() -> Vocab:
    return Vocab(load_config("config/config.yaml").tokenizer)


def _encode(vocab: Vocab, event_type: EventType, value: int = 0) -> int:
    return vocab.encode_event(Event(type=event_type, value=value))


def test_before_object_allows_abs_time_and_eos_only() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    mask = g.current_mask()
    start, end = vocab.token_range(EventType.ABS_TIME)
    assert mask[start:end].all()
    assert mask[int(SpecialToken.EOS)]
    circle_start, circle_end = vocab.token_range(EventType.CIRCLE)
    assert not mask[circle_start:circle_end].any()


def test_circle_flow() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    assert g.phase == GrammarPhase.HEADER_FRESH
    for t in (EventType.DISTANCE, EventType.POS, EventType.HITSOUND, EventType.VOLUME):
        g.update(_encode(vocab, t, 0))
        assert g.phase == GrammarPhase.HEADER_FRESH
    g.update(_encode(vocab, EventType.CIRCLE, 0))
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def test_slider_full_flow() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.DISTANCE, 0))
    g.update(_encode(vocab, EventType.POS, 10))
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    g.update(_encode(vocab, EventType.BEZIER_ANCHOR, 0))
    assert g.phase == GrammarPhase.NEED_ANCHOR_POS
    g.update(_encode(vocab, EventType.POS, 20))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    g.update(_encode(vocab, EventType.RED_ANCHOR, 0))
    assert g.phase == GrammarPhase.NEED_ANCHOR_POS
    g.update(_encode(vocab, EventType.POS, 30))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    g.update(_encode(vocab, EventType.ABS_TIME, 150))
    assert g.phase == GrammarPhase.SLIDER_END_HEADER
    g.update(_encode(vocab, EventType.DISTANCE, 0))
    g.update(_encode(vocab, EventType.POS, 40))
    g.update(_encode(vocab, EventType.LAST_ANCHOR, 0))
    assert g.phase == GrammarPhase.AFTER_LAST_ANCHOR
    g.update(_encode(vocab, EventType.SLIDER_END, 0))
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def test_spinner_flow() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.SPINNER, 0))
    assert g.phase == GrammarPhase.SPINNER_BODY
    g.update(_encode(vocab, EventType.ABS_TIME, 150))
    assert g.phase == GrammarPhase.SPINNER_END_HEADER
    g.update(_encode(vocab, EventType.SPINNER_END, 0))
    assert g.phase == GrammarPhase.BEFORE_OBJECT


def test_mask_in_slider_anchors_forbids_circle() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    mask = g.current_mask()
    circle_start, circle_end = vocab.token_range(EventType.CIRCLE)
    assert not mask[circle_start:circle_end].any()
    bez_start, bez_end = vocab.token_range(EventType.BEZIER_ANCHOR)
    assert mask[bez_start:bez_end].all()


def test_need_anchor_pos_only_allows_pos() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    g.update(_encode(vocab, EventType.BEZIER_ANCHOR, 0))
    mask = g.current_mask()
    pos_start, pos_end = vocab.token_range(EventType.POS)
    assert mask[pos_start:pos_end].all()
    bez_start, bez_end = vocab.token_range(EventType.BEZIER_ANCHOR)
    assert not mask[bez_start:bez_end].any()
    circle_start, circle_end = vocab.token_range(EventType.CIRCLE)
    assert not mask[circle_start:circle_end].any()


def test_after_last_anchor_allows_slides_and_end_only() -> None:
    vocab = _vocab()
    g = GrammarState(vocab)
    g.phase = GrammarPhase.AFTER_LAST_ANCHOR
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
    g.update(_encode(vocab, EventType.ABS_TIME, 100))
    g.update(_encode(vocab, EventType.SLIDER_HEAD, 0))
    assert g.phase == GrammarPhase.IN_SLIDER_ANCHORS
    g.reset()
    assert g.phase == GrammarPhase.BEFORE_OBJECT
