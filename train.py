from src.utils.config import load_config
from src.training import TrainingSession
import json
config = load_config("./config.yaml")

training_session = TrainingSession(config)
print("Starting training session...")
print("Session path:", training_session.path)


training_session._load_files_path()
print(f"Number of files in dataset: {len(training_session.files)}")
print(json.dumps(training_session.file_dict, indent=2))