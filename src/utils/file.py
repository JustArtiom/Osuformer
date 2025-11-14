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
