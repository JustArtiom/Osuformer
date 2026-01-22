import sys
import subprocess
from pathlib import Path
import os
from src.utils import difflog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = PROJECT_ROOT / ".gitignore"
REQ_FILE = PROJECT_ROOT / "requirements.txt"
TMP_FILE = PROJECT_ROOT / ".requirements.txt"


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

  return sorted([line.split("==")[0].lower() for line in lines])


def _run_pipreqs(ignored):
  cmd = [
    sys.executable,
    "-m",
    "pipreqs.pipreqs",
    str(PROJECT_ROOT),
    "--force",
    "--savepath",
    str(TMP_FILE)
  ]

  if ignored:
    cmd += ["--ignore", ",".join(ignored)]

  subprocess.check_call(cmd)


def requirements_up_to_date() -> tuple[bool, str, str]:
  ignored = _read_gitignore_dirs()

  current = _normalized_requirements(REQ_FILE)
  _run_pipreqs(ignored)
  generated = _normalized_requirements(TMP_FILE)
  os.remove(TMP_FILE)

  current_set = set(current)
  generated_set = set(generated)

  missing = sorted(generated_set - current_set)

  up_to_date = len(missing) == 0

  return up_to_date, "\n".join(current), "\n".join(generated)

def compute_package_version(package: str) -> str:
  try:
    import pkg_resources
    version = pkg_resources.get_distribution(package).version
    return version
  except Exception:
    return "unknown"

def main():
  up_to_date, current, generated = requirements_up_to_date()
  if(up_to_date):
    print("Requirements are up to date.")
    return

  print(difflog(
    current,
    generated,
  ))

  print(
    "Recommended to update requirements.txt.\n------------", 
    "\n".join([f"{pkg}=={compute_package_version(pkg)}" for pkg in generated.splitlines()]), 
    "\n------------"
  )
if __name__ == "__main__":
  main()
