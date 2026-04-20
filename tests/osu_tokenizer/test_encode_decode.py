from src.osu_tokenizer import SpecialToken, Vocab, attach_rel_times, beatmap_to_events, decode, encode

from .fixtures import build_beatmap, make_circle, make_config, make_slider, make_timing_point


def test_roundtrip_full_pipeline() -> None:
    cfg = make_config()
    vocab = Vocab(cfg)
    beatmap = build_beatmap(
        [
            make_circle(time=100.0),
            make_slider(time=500.0, head=(100.0, 100.0), anchors=[(200.0, 200.0)]),
            make_circle(time=1200.0, x=400.0, y=300.0),
        ],
        [make_timing_point()],
    )
    stream = beatmap_to_events(beatmap, window_start_ms=100.0, vocab=vocab, config=cfg)
    with_rel = attach_rel_times(stream.events, vocab, cfg)
    prefixed = [SpecialToken.SOS, *with_rel, SpecialToken.EOS]
    token_ids = encode(prefixed, vocab)
    decoded = decode(token_ids, vocab)
    assert decoded == prefixed
