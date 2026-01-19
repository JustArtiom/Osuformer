def fmt(n:float) -> str:
  if n.is_integer():
    return f"{int(n)}"
  else:
    return f"{n}".rstrip("0").rstrip(".")