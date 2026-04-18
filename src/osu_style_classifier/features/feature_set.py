from dataclasses import dataclass, field


@dataclass
class TimingFeatures:
    primary_bpm: float = 0.0
    bpm_min: float = 0.0
    bpm_max: float = 0.0
    bpm_count: int = 0
    uninherited_count: int = 0
    inherited_count: int = 0
    meters: list[int] = field(default_factory=list)
    non_4_4_ratio: float = 0.0
    sv_multipliers: list[float] = field(default_factory=list)
    sv_changes_per_minute: float = 0.0
    avg_sv_multiplier: float = 1.0
    max_sv_multiplier: float = 1.0
    min_sv_multiplier: float = 1.0


@dataclass
class SnapFeatures:
    divisor_counts: dict[int, int] = field(default_factory=dict)
    unsnapped_count: int = 0
    total_intervals: int = 0
    diversity: int = 0
    entropy_bits: float = 0.0
    complex_divisor_count: int = 0


@dataclass
class ObjectFeatures:
    total: int = 0
    circles: int = 0
    sliders: int = 0
    spinners: int = 0
    spinner_durations_ms: list[float] = field(default_factory=list)
    object_density_per_second: float = 0.0
    peak_density_per_second: float = 0.0
    drain_time_seconds: float = 0.0
    total_length_seconds: float = 0.0
    distances_px: list[float] = field(default_factory=list)
    flow_angles_deg: list[float] = field(default_factory=list)
    avg_distance_px: float = 0.0
    std_distance_px: float = 0.0
    avg_flow_angle_deg: float = 0.0
    std_flow_angle_deg: float = 0.0
    simultaneous_object_count: int = 0


@dataclass
class StreamFeatures:
    stream_runs: list[int] = field(default_factory=list)
    stream_run_distances_px: list[float] = field(default_factory=list)
    max_run_length: int = 0
    stream_object_ratio: float = 0.0
    burst_count: int = 0
    cutstream_count: int = 0
    spaced_stream_count: int = 0
    longest_sustained_stream_seconds: float = 0.0


@dataclass
class SliderFeatures:
    avg_length_px: float = 0.0
    avg_anchors: float = 0.0
    multi_curve_ratio: float = 0.0
    short_slider_ratio: float = 0.0
    max_anchors: int = 0


@dataclass
class SpatialFeatures:
    coverage_ratio: float = 0.0
    grid_snap_ratio: float = 0.0
    symmetry_x_score: float = 0.0
    symmetry_y_score: float = 0.0
    stack_pairs: int = 0
    perfect_stack_count: int = 0
    overlap_ratio: float = 0.0


@dataclass
class BeatmapFeatures:
    circle_size: float = 5.0
    approach_rate: float = 5.0
    timing: TimingFeatures = field(default_factory=TimingFeatures)
    snap: SnapFeatures = field(default_factory=SnapFeatures)
    objects: ObjectFeatures = field(default_factory=ObjectFeatures)
    streams: StreamFeatures = field(default_factory=StreamFeatures)
    sliders: SliderFeatures = field(default_factory=SliderFeatures)
    spatial: SpatialFeatures = field(default_factory=SpatialFeatures)
    storyboard_event_count: int = 0
    has_video: bool = False
    custom_sample_ratio: float = 0.0
