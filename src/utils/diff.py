from difflib import unified_diff, SequenceMatcher
from difflib import SequenceMatcher
import colorama
colorama.init()

def diff(a: str, b: str) -> str:
  lines = unified_diff(
    a.splitlines(keepends=True),
    b.splitlines(keepends=True),
    fromfile="expected",
    tofile="parsed",
    lineterm=""
  )

  output = []

  for line in lines:
    if line.startswith("+") and not line.startswith("+++"):
      output.append(f"\033[32m{line}\033[0m")
    elif line.startswith("-") and not line.startswith("---"):
      output.append(f"\033[31m{line}\033[0m")
    elif line.startswith("@@"):
      output.append(f"\033[36m{line}\033[0m")
    else:
      output.append(line)

  return "".join(output)

def similarity(a: str, b: str) -> float:
  return SequenceMatcher(None, a.splitlines(), b.splitlines()).ratio() * 100