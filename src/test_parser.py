from osu import (
    Beatmap,
    calculate_difficulty,
    calculate_performance,
)


beatmap = Beatmap(file_path="dataset/new beginnings - example map.osu")
difficulty = calculate_difficulty(beatmap)
performance = calculate_performance(beatmap, difficulty)
print(f"star_rating={difficulty.star_rating:.4f}")
print(f"aim={difficulty.aim:.4f} speed={difficulty.speed:.4f}")
print(
    "pp="
    f"{performance.pp:.2f} (aim={performance.aim_pp:.2f}, "
    f"speed={performance.speed_pp:.2f}, acc={performance.accuracy_pp:.2f})"
)
