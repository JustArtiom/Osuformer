from src.osu_tokenizer import Event, EventType, SPECIAL_COUNT, SpecialToken, Vocab

from .fixtures import make_config


def test_vocab_ranges_are_contiguous() -> None:
    vocab = Vocab(make_config())
    offset = SPECIAL_COUNT
    for er in vocab.output_ranges:
        start, end = vocab.token_range(er.type)
        assert start == offset
        assert end == offset + er.size
        offset = end
    assert offset == vocab.vocab_size_out
    for er in vocab.input_ranges:
        start, end = vocab.token_range(er.type)
        assert start == offset
        assert end == offset + er.size
        offset = end
    assert offset == vocab.vocab_size_in


def test_specials_occupy_first_ids() -> None:
    vocab = Vocab(make_config())
    for st in SpecialToken:
        decoded = vocab.decode_token(int(st))
        assert isinstance(decoded, SpecialToken)
        assert decoded == st


def test_encode_decode_roundtrip_for_every_output_range() -> None:
    vocab = Vocab(make_config())
    for er in vocab.output_ranges:
        for value in {er.min_value, (er.min_value + er.max_value) // 2, er.max_value}:
            ev = Event(type=er.type, value=value)
            token_id = vocab.encode_event(ev)
            decoded = vocab.decode_token(token_id)
            assert decoded == ev


def test_encode_decode_roundtrip_for_every_input_range() -> None:
    vocab = Vocab(make_config())
    for er in vocab.input_ranges:
        for value in {er.min_value, (er.min_value + er.max_value) // 2, er.max_value}:
            ev = Event(type=er.type, value=value)
            token_id = vocab.encode_event(ev)
            decoded = vocab.decode_token(token_id)
            assert decoded == ev


def test_position_encode_decode_roundtrip() -> None:
    vocab = Vocab(make_config())
    x, y = 256.0, 192.0
    value = vocab.grid.encode(x, y)
    rx, ry = vocab.grid.decode(value)
    assert abs(rx - x) <= vocab.config.coordinate_step
    assert abs(ry - y) <= vocab.config.coordinate_step


def test_value_out_of_range_raises() -> None:
    import pytest
    vocab = Vocab(make_config())
    ar = vocab.range_for(EventType.ABS_TIME)
    with pytest.raises(ValueError):
        vocab.encode_event(Event(type=EventType.ABS_TIME, value=ar.max_value + 1))
