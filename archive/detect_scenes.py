#!/usr/bin/env python3
"""
Определяем конец заставки как конец самого длинного префикса,
в котором ≥ 50 % всех КОМБИНАЦИЙ эпизодов имеют похожий кадр
(Hamming(pHash) ≤ 16).  Допускаем до 5 «плохих» кадров подряд.
"""
from pathlib import Path
from PIL import Image
import imagehash, numpy as np, itertools, json
from tqdm import tqdm

FRAMES_ROOT = Path("../data/frames")  # кадры 2 fps
OUT_DIR = Path("data/intro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 1
SIM_THRESH = 20  # pHash ≤ 16 → «похоже»
PAIR_SHARE_MIN = 0.40  # ≥50 % пар совпадают
GAP_ALLOW = 10  # допустимый провал (кадры)
MAX_FRAMES = 1200  # обрабатываем первые 10 минут  (safety)


def phash_seq(frame_dir):
    frames = sorted(frame_dir.glob("*.jpg"))[:MAX_FRAMES]
    arr = np.zeros(len(frames), dtype=np.uint64)
    for i, fp in enumerate(frames):
        arr[i] = int(str(imagehash.phash(Image.open(fp))), 16)
    return arr


def detect_intro(seqs):
    n = len(seqs)
    L = min(len(s) for s in seqs)
    pair_idx = list(itertools.combinations(range(n), 2))
    pair_cnt = len(pair_idx)

    best_idx, streak = 0, 0
    for idx in range(L):
        match_pairs = sum(
            bin(seqs[a][idx] ^ seqs[b][idx]).count("1") <= SIM_THRESH
            for a, b in pair_idx
        )
        share = match_pairs / pair_cnt
        if share >= PAIR_SHARE_MIN:
            best_idx = idx
            streak = 0
        else:
            streak += 1
            if streak > GAP_ALLOW:
                break
    return (best_idx + 1) / FPS


def main():
    for show_dir in sorted(FRAMES_ROOT.iterdir()):
        if not show_dir.is_dir():
            continue
        eps = sorted(p for p in show_dir.iterdir() if p.is_dir())
        if not eps:
            continue
        print(f"[{show_dir.name}] episodes: {len(eps)}")

        seqs = [phash_seq(ep) for ep in tqdm(eps, desc="hashing")]
        intro_end_sec = detect_intro(seqs)

        print(f"→ common intro ≈ {intro_end_sec:.2f} s")

        # сохраняем одинаковый таймкод для всех эпизодов шоу
        for ep in eps:
            out_json = OUT_DIR / f"{ep.name}.json"
            json.dump({"episode": ep.name,
                       "pred_start_main": intro_end_sec},
                      out_json.open("w"), indent=2)


if __name__ == "__main__":
    main()
