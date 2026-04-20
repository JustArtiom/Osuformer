from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    version: str
    dt_bin_ms: int
    context_ms: int
    generate_ms: int
    lookahead_ms: int
    coordinate_step: int
    coordinate_padding: int
    distance_max_px: int
    snap_max: int
    hitsound_count: int
    volume_max: int
    slider_slides_max: int
    scroll_speed_max: int
    difficulty_bins: int
    difficulty_max_star: float
    cs_step: float
    ar_step: float
    od_step: float
    hp_step: float
    stat_max: float
    year_min: int
    year_max: int
    song_length_buckets: int
    song_length_bucket_s: int
    song_position_min: int
    song_position_max: int
    global_sv_min: int
    global_sv_max: int
    history_event_budget: int
