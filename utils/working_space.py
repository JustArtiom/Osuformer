from pathlib import Path


class WorkingSpace:
    def __init__(self, path: str):
        self.path = Path(path)

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at: {self.path}")
        else:
            print(f"Directory already exists at: {self.path}")
