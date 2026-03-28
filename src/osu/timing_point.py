from .enums import SampleSet, Effects


class TimingPoint:
    def __init__(
        self,
        *,
        raw: str = "",
        time: float = 0,
        beat_length: float = 500.0,
        meter: int = 4,
        sample_set: SampleSet = SampleSet.DEFAULT,
        sample_index: int = 0,
        volume: int = 100,
        uninherited: int = 1,
        effects: Effects = Effects.NONE,
    ):
        self.time = time
        self.beat_length = beat_length
        self.meter = meter
        self.sample_set = sample_set
        self.sample_index = sample_index
        self.volume = volume
        self.uninherited = uninherited
        self.effects = effects

        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        segments = [s.strip() for s in raw.split(",")]
        self.time = float(segments[0])
        self.beat_length = float(segments[1])
        self.meter = int(segments[2]) if len(segments) > 2 else 4
        self.sample_set = SampleSet(int(segments[3])) if len(segments) > 3 else SampleSet.DEFAULT
        self.sample_index = int(segments[4]) if len(segments) > 4 else 0
        self.volume = int(segments[5]) if len(segments) > 5 else 100
        self.uninherited = int(segments[6]) if len(segments) > 6 else 1
        self.effects = Effects(int(segments[7])) if len(segments) > 7 else Effects.NONE

    @property
    def is_uninherited(self) -> bool:
        return self.uninherited == 1

    @property
    def is_kiai(self) -> bool:
        return bool(self.effects & Effects.KIAI)

    @property
    def is_omit_first_bar_line(self) -> bool:
        return bool(self.effects & Effects.OMIT_FIRST_BAR_LINE)

    def get_bpm(self) -> float:
        if self.uninherited == 0:
            raise ValueError("Cannot get BPM from an inherited timing point")
        if self.beat_length <= 0:
            raise ValueError(f"Invalid beat_length for uninherited timing point: {self.beat_length}")
        return 60000.0 / self.beat_length

    def get_slider_velocity_multiplier(self) -> float:
        if self.uninherited == 1:
            return 1.0
        return -100.0 / self.beat_length

    def _format_beat_length(self, value: float) -> str:
        result = format(value, ".15g")
        if result == "-0":
            return "0"
        return result

    def __str__(self) -> str:
        beat_length = self._format_beat_length(self.beat_length)
        time = int(self.time) if self.time == int(self.time) else self.time
        return f"{time},{beat_length},{self.meter},{int(self.sample_set)},{self.sample_index},{self.volume},{self.uninherited},{int(self.effects)}"
