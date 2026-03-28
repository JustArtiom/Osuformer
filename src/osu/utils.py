def fmt(n: float) -> str:
    if n != n or n == float("inf") or n == float("-inf"):
        return "0"
    if n == int(n):
        return str(int(n))
    s = repr(n)
    if "e" in s or "E" in s:
        # Convert scientific notation to fixed-point, strip trailing zeros
        return f"{n:.15f}".rstrip("0").rstrip(".")
    return s
