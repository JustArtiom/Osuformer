import subprocess
import sys
import hashlib
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


def _file_hash(path: Path):
  if not path.exists():
    return None

  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)

  return h.hexdigest()


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

  before = _file_hash(REQ_FILE)

  _run_pipreqs(ignored)

  after = _file_hash(REQ_FILE)

  return before == after


def main():
  updated = requirements_up_to_date()
  print("Requirements are up to date." if updated else "Requirements were updated.")

if __name__ == "__main__":
  main()