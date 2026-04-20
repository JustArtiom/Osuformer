from src.osu.enums import HitSound, SampleSet
from src.osu_tokenizer import decode_hitsound, encode_hitsound


def test_normal_hit_default_sets() -> None:
    value = encode_hitsound(HitSound.NORMAL, SampleSet.NORMAL, SampleSet.NORMAL)
    assert 0 <= value <= 72
    hs, ns, as_ = decode_hitsound(value)
    assert ns == SampleSet.NORMAL
    assert as_ == SampleSet.NORMAL


def test_whistle_soft_drum_encoding() -> None:
    value = encode_hitsound(HitSound.WHISTLE, SampleSet.SOFT, SampleSet.DRUM)
    hs, ns, as_ = decode_hitsound(value)
    assert ns == SampleSet.SOFT
    assert as_ == SampleSet.DRUM


def test_encode_never_exceeds_range() -> None:
    for hs_bits in range(0, 16):
        for ns in (SampleSet.NORMAL, SampleSet.SOFT, SampleSet.DRUM):
            for as_ in (SampleSet.NORMAL, SampleSet.SOFT, SampleSet.DRUM):
                value = encode_hitsound(HitSound(hs_bits), ns, as_)
                assert 0 <= value <= 72
