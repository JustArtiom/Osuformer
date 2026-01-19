from scripts.requirements import requirements_up_to_date


def test_requirements_txt_is_up_to_date():
  assert requirements_up_to_date(), (
    "requirements.txt is outdated. "
    "Run python scripts/requirements.py and commit the updated file."
  )