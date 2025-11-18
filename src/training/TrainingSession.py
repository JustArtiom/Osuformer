import os

class TrainingSession:
  def __init__(self, config, resume_path: str | None = None):
    self.config = config
    self.dataset_path = config["paths"]["data"]
    self.base_path = os.path.abspath(config["training"]["path"])
    os.makedirs(self.base_path, exist_ok=True)
    self.created = False

    if resume_path:
      self.path = os.path.abspath(resume_path)
      if not os.path.isdir(self.path):
        raise FileNotFoundError(f"Resume directory '{self.path}' does not exist.")
      self.training_name = os.path.basename(self.path.rstrip(os.sep))
      self.created = True
    else:
      self.training_name = TrainingSession.auto(self.base_path)
      self.path = os.path.join(self.base_path, self.training_name)

  @staticmethod
  def auto(path):
    existing = os.listdir(path)
    max_num = 0
    for name in existing:
      if name.isdigit():
        num = int(name)
        if num > max_num:
          max_num = num
    return str(max_num + 1)

  def ensure_created(self):
    if not self.created:
      os.makedirs(self.path, exist_ok=True)
      self.created = True
