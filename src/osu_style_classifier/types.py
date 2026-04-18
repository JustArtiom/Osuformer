from dataclasses import dataclass, field

from .tags import StyleTag


@dataclass(frozen=True)
class TagPrediction:
    tag: StyleTag
    confidence: float


@dataclass
class StyleResult:
    predictions: list[TagPrediction] = field(default_factory=list)

    @property
    def tags(self) -> list[StyleTag]:
        return [p.tag for p in self.predictions]

    def filter(self, min_confidence: float) -> "StyleResult":
        return StyleResult(
            predictions=[p for p in self.predictions if p.confidence >= min_confidence]
        )
