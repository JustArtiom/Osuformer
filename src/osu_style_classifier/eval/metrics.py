from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TagMetrics:
    tag: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative


@dataclass
class EvalReport:
    per_tag: dict[str, TagMetrics] = field(default_factory=dict)
    sample_count: int = 0
    labeled_count: int = 0

    def update(self, predicted: set[str], actual: set[str]) -> None:
        for tag in predicted | actual:
            m = self.per_tag.setdefault(tag, TagMetrics(tag=tag))
            in_pred = tag in predicted
            in_actual = tag in actual
            if in_pred and in_actual:
                m.true_positive += 1
            elif in_pred and not in_actual:
                m.false_positive += 1
            elif not in_pred and in_actual:
                m.false_negative += 1

    def macro_f1(self) -> float:
        if not self.per_tag:
            return 0.0
        return sum(m.f1 for m in self.per_tag.values()) / len(self.per_tag)

    def summary(self) -> str:
        rows = sorted(self.per_tag.values(), key=lambda m: (-m.support, m.tag))
        lines = [
            f"Samples: {self.sample_count}  Labeled: {self.labeled_count}  Macro F1: {self.macro_f1():.3f}",
            f"{'tag':<38} {'prec':>6} {'rec':>6} {'f1':>6} {'sup':>5} {'tp':>4} {'fp':>4} {'fn':>4}",
        ]
        for m in rows:
            lines.append(
                f"{m.tag:<38} {m.precision:>6.2f} {m.recall:>6.2f} {m.f1:>6.2f} {m.support:>5d} {m.true_positive:>4d} {m.false_positive:>4d} {m.false_negative:>4d}"
            )
        return "\n".join(lines)
