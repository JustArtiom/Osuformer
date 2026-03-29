from src.osu.utils import fmt


class TestFmt:
    def test_integer_float(self):
        assert fmt(100.0) == "100"
        assert fmt(0.0) == "0"
        assert fmt(-5.0) == "-5"

    def test_decimal(self):
        assert fmt(1.5) == "1.5"
        assert fmt(0.001) == "0.001"

    def test_no_scientific_notation(self):
        assert "e" not in fmt(1e-7).lower()
        assert fmt(1e-7) == "0.0000001"

    def test_high_precision_preserved(self):
        assert fmt(135.000005149842) == "135.000005149842"

    def test_negative_zero(self):
        assert fmt(-0.0) == "0"

    def test_nan_returns_zero(self):
        assert fmt(float("nan")) == "0"

    def test_inf_returns_zero(self):
        assert fmt(float("inf")) == "0"
        assert fmt(float("-inf")) == "0"
