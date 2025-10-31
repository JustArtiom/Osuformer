import os
from ..utils.file import collect_files, build_file_tree, dictify_files
from ..audio import AUDIO_EXTENSIONS

class TrainingSession:
  def __init__(self, config):
    self.config = config
    self.dataset_path = config["paths"]["data"]
    self.path = os.path.join(config["training"]["path"])
    if not os.path.exists(self.path):
      os.makedirs(self.path, exist_ok=True)

    self.training_name = TrainingSession.auto(self.path)
    self.path = os.path.join(self.path, self.training_name)

    if not os.path.exists(self.path):
      os.makedirs(self.path, exist_ok=True)

  @staticmethod
  def auto(path):
    existing = os.listdir(f"./{path}")
    max_num = 0
    for name in existing:
      if name.isdigit():
        num = int(name)
        if num > max_num:
          max_num = num
    return str(max_num + 1)
  
  def _load_files_path(self):
    self.files = collect_files(self.dataset_path, [*AUDIO_EXTENSIONS, "*.mel.npz", "*.osu"])
    self.file_tree = build_file_tree(self.files)
    self.file_dict = dictify_files(self.file_tree)

