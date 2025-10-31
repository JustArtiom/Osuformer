from collections import defaultdict
import os
import glob
import fnmatch

def collect_files(base_dir, include_patterns=["*"], exclude_patterns=[]):
    all_files = []
    for pattern in include_patterns:
        all_files.extend(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))

    filtered_files = []
    for file_path in all_files:
        if not any(fnmatch.fnmatch(os.path.basename(file_path), pat) for pat in exclude_patterns):
            rel_path = os.path.relpath(file_path, base_dir)
            filtered_files.append(rel_path)

    return filtered_files


def tree():
    return defaultdict(tree)

def insert_path(d, path):
    parts = path.split(os.sep)
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = None

def build_file_tree(paths):
    root = tree()
    for p in paths:
        insert_path(root, p)
    return root

def dictify_files(d):
    if isinstance(d, defaultdict):
        if all(v is None for v in d.values()):
            return list(d.keys())
        return {k: dictify_files(v) for k, v in d.items()}
    return d

AUDIO_EXTENSIONS = ["*.wav", "*.mp3", "*.ogg", "*.flac"]