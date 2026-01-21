from tqdm import tqdm
from typing import List, Union
import shutil, os
from pathlib import Path

def mkdir(path: Union[str, Path], overwrite: bool = False) -> Path:
  if overwrite and os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path, exist_ok=True)
  return Path(path) if isinstance(path, str) else path