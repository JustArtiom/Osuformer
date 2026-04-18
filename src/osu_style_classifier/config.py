from dataclasses import dataclass, field


@dataclass
class StreamConfig:
    min_bpm: float = 160.0
    snap_tolerance_ms: float = 8.0
    min_run_length: int = 10
    burst_min_length: int = 3
    burst_max_length: int = 9
    spaced_px_per_ms: float = 1.8
    cutstream_max_gap_beats: float = 1.5
    max_stream_interval_ms: float = 140.0
    burst_count_threshold: int = 25
    cutstream_count_threshold: int = 40
    stamina_min_stream_ratio: float = 0.35
    burst_min_bpm: float = 150.0


@dataclass
class AimConfig:
    jump_min_avg_distance_px: float = 140.0
    spaced_jumps_min_avg_px: float = 220.0
    wide_min_avg_angle_deg: float = 120.0
    linear_max_angle_std_deg: float = 35.0
    linear_max_avg_angle_deg: float = 55.0
    sharp_min_avg_angle_deg: float = 115.0
    sharp_max_angle_std_deg: float = 60.0
    aim_control_min_angle_std_deg: float = 60.0
    aim_control_min_avg_distance_px: float = 140.0
    aim_control_min_total_objects: int = 250
    flow_max_angle_std_deg: float = 45.0
    flow_min_avg_angle_deg: float = 70.0
    flow_max_avg_angle_deg: float = 140.0
    precision_min_cs: float = 4.5
    precision_max_spacing_px: float = 90.0
    stamina_min_objects: int = 700
    stamina_min_drain_seconds: float = 240.0
    stamina_min_avg_distance_px: float = 180.0
    alt_min_bpm: float = 185.0
    alt_min_density: float = 5.5
    alt_min_avg_distance_px: float = 90.0
    alt_max_avg_distance_px: float = 260.0
    alt_min_half_beat_ratio: float = 0.35


@dataclass
class SliderConfig:
    slider_only_min_ratio: float = 0.70
    circle_only_min_ratio: float = 0.85
    slider_tech_min_anchors_per_slider: float = 8.0
    slider_tech_min_short_slider_ratio: float = 0.35
    slidershapes_min_avg_length_px: float = 220.0
    slidershapes_min_multi_curve_ratio: float = 0.30
    high_sv_multiplier: float = 1.8
    low_sv_multiplier: float = 0.6
    complex_sv_changes_per_minute: float = 40.0
    complex_sv_min_range: float = 0.8


@dataclass
class RhythmConfig:
    simple_max_snap_diversity: int = 2
    chaotic_min_snap_diversity: int = 6
    chaotic_min_entropy_bits: float = 2.5
    finger_control_min_snap_diversity: int = 5
    finger_control_min_density: float = 4.5
    variable_timing_min_uninherited: int = 40
    time_signatures_min_non_4_4_ratio: float = 0.35


@dataclass
class VisualConfig:
    grid_snap_coord_step: float = 4.0
    grid_snap_min_ratio: float = 0.95
    symmetry_min_axis_score: float = 0.70
    playfield_constraint_max_coverage: float = 0.30
    playfield_usage_min_coverage: float = 0.98
    perfect_stacks_min_count: int = 200
    storyboard_min_events: int = 50
    perfect_stacks_max_distance_px: float = 1.5
    visually_dense_min_concurrent: float = 15.0
    overlap_min_ratio: float = 0.30


@dataclass
class MetaConfig:
    ninja_spinner_max_ms: float = 800.0
    keysounds_min_custom_sample_ratio: float = 0.30
    two_b_min_overlap_ms: float = 1.0
    two_b_min_overlap_count: int = 20


@dataclass
class ClassifierConfig:
    streams: StreamConfig = field(default_factory=StreamConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    sliders: SliderConfig = field(default_factory=SliderConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    min_confidence: float = 0.5
