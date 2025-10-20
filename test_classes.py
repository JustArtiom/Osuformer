import os
from src import osu

def read_folders(dataset_path):
    class_folders = []
    for entry in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, entry)
        if os.path.isdir(full_path):
            class_folders.append(entry)
    return class_folders

classes = read_folders('dataset/classes')
for cls in classes:
    print(f"Class: {cls}")
    maps = read_folders(os.path.join('dataset/classes', cls))
    for m in maps:
        print(f"  Map: {m}")
        files = os.listdir(os.path.join('dataset/classes', cls, m))
        for f in files:
            if f.endswith('.osu'):
                print(f"    File: {f}")
                osu_file_path = os.path.join('dataset/classes', cls, m, f)
                beatmap = osu.Beatmap(file_path=osu_file_path)
                difficulty = beatmap.get_difficulty()
                performance = beatmap.get_performance()
                print(f"        star_rating={difficulty.star_rating:.4f}")
                print(f"        aim={difficulty.aim:.4f} speed={difficulty.speed:.4f}")
                print(
                    "        pp="
                    f"{performance.pp:.2f} (aim={performance.aim_pp:.2f}, "
                    f"speed={performance.speed_pp:.2f}, acc={performance.accuracy_pp:.2f})"
                )

                print(f"        len of sliders = {len([obj for obj in beatmap.hit_objects if isinstance(obj, osu.Slider)])}")
                print(f"        len of circles = {len([obj for obj in beatmap.hit_objects if isinstance(obj, osu.Circle)])}")