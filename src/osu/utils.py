def fmt(n: float) -> str:
    if n != n or n == float("inf") or n == float("-inf"):
        return "0"
    if n == int(n):
        return str(int(n))
    result = f"{n:.10f}".rstrip("0").rstrip(".")
    return "0" if result == "-0" else result
