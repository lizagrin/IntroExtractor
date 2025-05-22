import json, librosa, numpy as np, pathlib
from scipy.signal import correlate

DATA_DIR = pathlib.Path("../data/audio")


def chroma_vec(wav, sr):
    # CQT-chroma, hop = 0.5 s
    hop = int(0.5 * sr)
    C = librosa.feature.chroma_cqt(y=wav, sr=sr, hop_length=hop)
    return librosa.util.normalize(C, axis=0)


# -------- 1. строим/загружаем эталон -------------------------------
fp_file = pathlib.Path("fingerprints.json")
if fp_file.exists():
    fingerprints = json.load(fp_file)
else:
    fingerprints = {}
    for show_dir in DATA_DIR.iterdir():
        first = sorted(show_dir.glob("*.wav"))[0]
        y, sr = librosa.load(first, sr=48000, mono=True, duration=35)
        fingerprints[show_dir.name] = chroma_vec(y, sr).mean(axis=1).tolist()
    fp_file.write_text(json.dumps(fingerprints))

# -------- 2. обходим все ep ---------------------------------------
results = []
for wav_path in DATA_DIR.rglob("*.wav"):
    show = wav_path.parent.name  # show1 / show2
    fp = np.array(fingerprints[show])  # (12,)

    y, sr = librosa.load(wav_path, sr=48000, mono=True)
    C = chroma_vec(y, sr)  # (12, T)

    # скользящее окно 35 s (~70 хопов)
    w = 70
    sims = [np.dot(fp, C[:, i:i + w].mean(axis=1))
            for i in range(0, C.shape[1] - w)]
    sims = np.array(sims)

    start_sec = np.argmax(sims) * 0.5
    sims_end = sims[::-1]
    end_sec = (len(sims) - np.argmax(sims_end) - w) * 0.5

    results.append((wav_path.name, round(start_sec), round(end_sec)))

# -------- 3. вывод -------------------------------------------------
for f, s, e in results:
    print(f"{f:50}  intro≈{s:4d}s   credits≈{e:4d}s")
