import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = PROJECT_ROOT / ".gitignore"
REQ_FILE = PROJECT_ROOT / "requirements.txt"


def _read_gitignore_dirs():
  if not GITIGNORE.exists():
    return []

  ignored = []

  for line in GITIGNORE.read_text().splitlines():
    line = line.strip()

    if not line or line.startswith("#"):
      continue

    if line.endswith("/"):
      ignored.append(line.rstrip("/"))

  return ignored


def _normalized_requirements(path: Path) -> list[str]:
  if not path.exists():
    return []

  lines = []
  for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    lines.append(line)

  return sorted(lines)


def _run_pipreqs(ignored):
  cmd = [
    sys.executable,
    "-m",
    "pipreqs.pipreqs",
    str(PROJECT_ROOT),
    "--force"
  ]

  if ignored:
    cmd += ["--ignore", ",".join(ignored)]

  subprocess.check_call(cmd)


def requirements_up_to_date() -> bool:
  ignored = _read_gitignore_dirs()

  current = _normalized_requirements(REQ_FILE)
  _run_pipreqs(ignored)
  generated = _normalized_requirements(REQ_FILE)

  return current == generated, "\n".join(current), "\n".join(generated)


def main():
  updated = requirements_up_to_date()
  print("Requirements are up to date." if updated else "Requirements were updated.")

if __name__ == "__main__":
  main()