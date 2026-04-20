from __future__ import annotations

from src.osu.enums import HitSound, SampleSet


def encode_hitsound(hit_sound: HitSound, normal_set: SampleSet, addition_set: SampleSet) -> int:
    bits = int(hit_sound) & 0b1111
    if bits & int(HitSound.NORMAL):
        bits_no_normal = bits & 0b1110
    else:
        bits_no_normal = bits
    hs_idx = bits_no_normal >> 1
    ns = max(1, int(normal_set)) if int(normal_set) > 0 else 1
    as_ = max(1, int(addition_set)) if int(addition_set) > 0 else 1
    return hs_idx + 8 * (ns - 1) + 24 * (as_ - 1)


def decode_hitsound(value: int) -> tuple[HitSound, SampleSet, SampleSet]:
    addition_idx = value // 24
    rem = value % 24
    sample_idx = rem // 8
    hs_bits = (rem % 8) << 1
    hit_sound = HitSound(hs_bits if hs_bits != 0 else 1)
    sample_set = SampleSet(sample_idx + 1)
    addition_set = SampleSet(addition_idx + 1)
    return hit_sound, sample_set, addition_set
