import argparse

from dotenv import load_dotenv

from src.osu_api import OsuClient
from src.osu_api.types import Beatmap, Beatmapset, UserTag

load_dotenv()

_MODE_NAMES = {0: "osu!", 1: "Taiko", 2: "Catch", 3: "Mania"}


def _display(beatmapset: Beatmapset) -> None:
    tags = beatmapset["tags"].split() if beatmapset["tags"] else []

    print(f"\n{'=' * 60}")
    print(f"  {beatmapset['artist']} - {beatmapset['title']}")
    print(f"  Mapper : {beatmapset['creator']}")
    print(f"  Status : {beatmapset['status']}")
    print(f"  BPM    : {beatmapset['bpm']}")
    print(f"  Set ID : {beatmapset['id']}")
    print(f"{'=' * 60}")

    user_tags: list[UserTag] = beatmapset.get("related_tags", [])
    print(f"\nUSER TAGS ({len(user_tags)})")
    if user_tags:
        for t in user_tags:
            ruleset_id = t.get("ruleset_id")
            ruleset = f" [mode {ruleset_id}]" if ruleset_id is not None else ""
            print(f"  {t['name']}{ruleset}")
            print(f"    {t['description']}")
    else:
        print("  (none)")

    print(f"\nCREATOR TAGS ({len(tags)})")
    if tags:
        print("  " + "  ".join(tags))
    else:
        print("  (none)")

    beatmaps: list[Beatmap] = beatmapset.get("beatmaps", [])
    if beatmaps:
        sorted_maps = sorted(beatmaps, key=lambda b: b["difficulty_rating"])
        print(f"\nDIFFICULTIES ({len(sorted_maps)})")
        for bm in sorted_maps:
            mode = _MODE_NAMES.get(bm["mode_int"], bm["mode"])
            print(
                f"  [{mode:5}] {bm['difficulty_rating']:.2f}★  "
                f"CS{bm['cs']} AR{bm['ar']} OD{bm['accuracy']}  "
                f"{bm['total_length'] // 60}:{bm['total_length'] % 60:02d}  "
                f"— {bm['version']}"
            )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch osu! beatmapset info")
    parser.add_argument("id", type=int, help="Beatmapset ID")
    args = parser.parse_args()

    client = OsuClient()
    beatmapset = client.beatmaps.get_beatmapset(args.id)
    _display(beatmapset)


if __name__ == "__main__":
    main()
