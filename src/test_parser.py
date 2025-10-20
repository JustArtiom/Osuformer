from osu import (
    Beatmap,
    Slider,
    Circle,
    calculate_difficulty,
    calculate_performance,
)


beatmap = Beatmap(file_path="dataset/lexycat - glitter  (thenoname) [Extra Glitter].osu")
difficulty = calculate_difficulty(beatmap, [])
performance = calculate_performance(difficulty)
print(f"star_rating={difficulty.star_rating:.4f}")
print(f"aim={difficulty.aim:.4f} speed={difficulty.speed:.4f}")
print(
    "pp="
    f"{performance.pp:.2f} (aim={performance.aim_pp:.2f}, "
    f"speed={performance.speed_pp:.2f}, acc={performance.accuracy_pp:.2f})"
)

print(f"len of sliders = {len([obj for obj in beatmap.hit_objects if isinstance(obj, Slider)])}")
print(f"len of circles = {len([obj for obj in beatmap.hit_objects if isinstance(obj, Circle)])}")