from scripts.requirements import requirements_up_to_date
from src.utils import diff

def test_requirements_txt_is_up_to_date():
  valid, current, generated = requirements_up_to_date()
  assert valid, (
    "requirements.txt is not up to date with the imports in the codebase.\n"
    "Please run `pipreqs . --force --savepath requirements.txt` to update it.\n\n"
    f"Diff:\n{diff(current, generated)}\n\n"

  )