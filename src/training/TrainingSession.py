import os

class TrainingSession:
  def __init__(self, config, resume_path: str | None = None):
    self.config = config
    self.dataset_path = config["paths"]["data"]
    base_path = os.path.abspath(config["training"]["path"])
    os.makedirs(base_path, exist_ok=True)

    if resume_path:
      self.path = os.path.abspath(resume_path)
      if not os.path.isdir(self.path):
        raise FileNotFoundError(f"Resume directory '{self.path}' does not exist.")
      self.training_name = os.path.basename(self.path.rstrip(os.sep))
    else:
      self.training_name = TrainingSession.auto(base_path)
      self.path = os.path.join(base_path, self.training_name)
      if not os.path.exists(self.path):
        os.makedirs(self.path, exist_ok=True)

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
