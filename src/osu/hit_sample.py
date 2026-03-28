from .enums import SampleSet


class HitSample:
    def __init__(
        self,
        *,
        raw: str = "",
        normal_set: SampleSet = SampleSet.DEFAULT,
        addition_set: SampleSet = SampleSet.DEFAULT,
        index: int = 0,
        volume: int = 0,
        filename: str = "",
    ):
        self.normal_set = normal_set
        self.addition_set = addition_set
        self.index = index
        self.volume = volume
        self.filename = filename

        if raw:
            self._load_raw(raw)

    def _load_raw(self, raw: str) -> None:
        parts = raw.split(":")
        self.normal_set = SampleSet(int(parts[0])) if len(parts) > 0 and parts[0].strip() else SampleSet.DEFAULT
        self.addition_set = SampleSet(int(parts[1])) if len(parts) > 1 and parts[1].strip() else SampleSet.DEFAULT
        self.index = int(parts[2]) if len(parts) > 2 and parts[2].strip() else 0
        self.volume = int(parts[3]) if len(parts) > 3 and parts[3].strip() else 0
        self.filename = parts[4].strip() if len(parts) > 4 else ""

    def __str__(self) -> str:
        return f"{int(self.normal_set)}:{int(self.addition_set)}:{self.index}:{self.volume}:{self.filename}"

    def __repr__(self) -> str:
        return (
            f"HitSample(normal_set={self.normal_set!r}, addition_set={self.addition_set!r}, "
            f"index={self.index}, volume={self.volume}, filename={self.filename!r})"
        )
