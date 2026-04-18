import io
import zipfile
from pathlib import Path

from tqdm import tqdm

SRC = Path.home() / "Downloads" / "Osu Dataset"
DEST = Path("/Volumes/MySSD/Songs")
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".aac", ".m4a"}


def get_mode(osu_content: str) -> int:
    for line in osu_content.splitlines():
        if line.strip().startswith("Mode"):
            _, _, value = line.partition(":")
            return int(value.strip())
    return 0


def get_audio_filename(osu_content: str) -> str | None:
    for line in osu_content.splitlines():
        if line.strip().startswith("AudioFilename"):
            _, _, value = line.partition(":")
            return value.strip()
    return None


def process_osz(osz_path: Path) -> None:
    folder_name = osz_path.stem
    out_dir = DEST / folder_name

    if out_dir.exists():
        return

    try:
        with zipfile.ZipFile(osz_path) as zf:
            names = zf.namelist()
            osu_files = [n for n in names if Path(n).suffix.lower() == ".osu"]

            standard_osu: list[str] = []
            audio_filename: str | None = None

            for name in osu_files:
                content = zf.read(name).decode("utf-8", errors="ignore")
                if get_mode(content) != 0:
                    continue
                standard_osu.append(name)
                if audio_filename is None:
                    audio_filename = get_audio_filename(content)

            if not standard_osu:
                return

            out_dir.mkdir(parents=True, exist_ok=True)

            for name in standard_osu:
                (out_dir / Path(name).name).write_bytes(zf.read(name))

            if audio_filename:
                audio_lower = audio_filename.lower()
                match = next(
                    (n for n in names if Path(n).name.lower() == audio_lower),
                    next((n for n in names if Path(n).suffix.lower() in AUDIO_EXTENSIONS), None),
                )
                if match:
                    (out_dir / Path(match).name).write_bytes(zf.read(match))

    except zipfile.BadZipFile:
        print(f"  [SKIP] bad zip: {osz_path.name}")
    except Exception as e:
        print(f"  [FAIL] {osz_path.name}: {e}")


def main() -> None:
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return
    if not DEST.parent.exists():
        print(f"USB not found: {DEST.parent} — is MySSD mounted?")
        return

    DEST.mkdir(exist_ok=True)

    osz_files = sorted(SRC.rglob("*.osz"))
    print(f"Found {len(osz_files)} .osz files\n")

    with tqdm(osz_files, unit="map") as bar:
        for osz_path in bar:
            bar.set_description(osz_path.stem[:60])
            process_osz(osz_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
