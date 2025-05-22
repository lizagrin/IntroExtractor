# Разбивает все серии на JPEG‑кадры (2 fps) и извлекает mono‑wav 16 kHz.

import subprocess
from pathlib import Path
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- настройки ----------
VIDEO_ROOT = Path(".")  # где лежат show1/, show2/...
OUT_FRAMES = Path("data/frames")  # куда кладём кадры
OUT_AUDIO = Path("data/audio")  # куда кладём wav
FPS = 1
SAMPLE_RATE = 16_000
MAX_WORKERS = 4  # параллельные FFmpeg-процессы
LOG_FILE = "prepare_media.log"
# --------------------------------

logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")


def run_cmd(cmd: list[str]) -> None:
    """Запускает ffmpeg и выводит лог‑строку при ошибке."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg failed: %s", e.stderr.decode("utf-8")[:200])
        raise


def process_episode(path: Path) -> None:
    """Обрабатывает один видео‑файл."""
    show = path.parent.name  # show1 / show2 …
    ep_stem = path.stem  # имя файла без .mp4/.mkv

    # ---- кадры ----
    frames_dir = OUT_FRAMES / show / ep_stem
    if not frames_dir.exists():
        frames_dir.mkdir(parents=True, exist_ok=True)
        cmd_frames = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-r", str(FPS),
            f"{frames_dir}/%06d.jpg"
        ]
        run_cmd(cmd_frames)

    # ---- аудио ----
    audio_dir = OUT_AUDIO / show
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_file = audio_dir / f"{ep_stem}.wav"
    if not wav_file.exists():
        cmd_audio = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-vn",  # no video
            "-ac", "1",  # mono
            "-ar", str(SAMPLE_RATE),
            str(wav_file)
        ]
        run_cmd(cmd_audio)

    logging.info("✓ processed %s", path.relative_to(VIDEO_ROOT))


def find_videos(root: Path) -> list[Path]:
    exts = {".mp4", ".mkv", ".mov"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def main():
    videos = find_videos(VIDEO_ROOT)
    print(f"Found {len(videos)} video files")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(process_episode, v): v for v in videos}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"Error processing {futs[fut].name}: {e}")

    print("Done. Logs →", LOG_FILE)


if __name__ == "__main__":
    main()
