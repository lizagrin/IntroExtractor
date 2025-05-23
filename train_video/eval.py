import numpy as np, torch, pathlib, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
from train_video.model import ClipAttention60


# ---------- utils -------------------------------------------------
def hhmmss_to_sec(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mad(x):  # median absolute deviation
    m = np.median(x)
    return np.median(np.abs(x - m))


def key_npz(p):
    return "_".join(p.stem.split("_")[1:-1])  # ..._windows.npz -> id


def key_csv(p):
    return pathlib.Path(p).stem  # show/ep.mp4 -> id


def find_first_block(mask):  # ≥5 ед.  ➜ ≥10 нулей
    run1 = run0 = 0
    for i, m in enumerate(mask):
        if m:
            run1 += 1
            run0 = 0
        else:
            run0 += 1
            if run1 >= 5 and run0 >= 10:
                return i - run0 + 1
    return 0


def find_last_block(mask):
    run1 = run0 = 0
    pos = len(mask) - 1
    for i in range(len(mask) - 1, -1, -1):
        m = mask[i]
        if m:
            run1 += 1
            run0 = 0
        else:
            run0 += 1
            if run1 >= 5 and run0 >= 10:
                pos = i + run0 - 1
                break
    return pos


# ---------- 1. модель --------------------------------------------
model = ClipAttention60()
model.load_state_dict(torch.load("../weights/best_clip_attention60.pth", map_location="cpu"))
model.eval()

# ---------- 2. метки ---------------------------------------------
labels = pd.read_csv("../data/labels.csv")
labels["key"] = labels["file"].apply(key_csv)
labels = labels.set_index("key")

# ---------- 3. обход эпизодов ------------------------------------
root = pathlib.Path("../data/clip_windows")
res = []

for f in sorted(root.glob("*_windows.npz")):
    key = key_npz(f)
    if key not in labels.index:
        print(f"[!] {key} пропущен (нет метки)")
        continue

    data = np.load(f)
    windows = data["windows"]  # (N,60,512)
    starts = data["start_indices"]  # (N,)

    # --- добавим «хвост», если остался неполный сегмент -----------------
    total_secs = starts[-1] + windows.shape[1]
    labelled_end = hhmmss_to_sec(labels.loc[key, "end_main"])
    if labelled_end + 30 > total_secs:  # <30 с до конца
        zeros = np.zeros((60, 512), dtype=np.float32)  # фиктивное окно
        windows = np.concatenate([windows, zeros[None]], axis=0)

    # --- вероятности p(t) ----------------------------------------------
    p = []
    with torch.no_grad():
        for w in torch.from_numpy(windows):
            p.append(model(w.unsqueeze(0)).squeeze(0).numpy())
    p = np.concatenate(p)  # (T,)

    # --- сглаживание mean(3) + median(7) -------------------------------
    p = uniform_filter1d(p, size=3)
    p = median_filter(p, size=7)

    # --- адаптивные пороги ---------------------------------------------
    base = np.median(p)
    dev = mad(p)
    thr_s = base + 2 * dev
    thr_e = np.percentile(p, 70)  # мягче для титров

    mask_s = p > thr_s
    mask_e = p > thr_e

    # --- границы -------------------------------------------------------
    s_pred = find_first_block(mask_s)

    tail = int(len(mask_e) * 0.50)  # последние 50 %
    e_pred = tail + find_last_block(mask_e[tail:])

    # --- ground-truth --------------------------------------------------
    s_true = hhmmss_to_sec(labels.loc[key, "start_main"])
    e_true = hhmmss_to_sec(labels.loc[key, "end_main"])
    res.append((key, s_pred, s_true, e_pred, e_true, p, thr_s, thr_e))

# ---------- 4. метрики ----------------------------------------------
mae_s = np.mean([abs(p - s) for _, p, s, _, _, _, _, _ in res])
mae_e = np.mean([abs(p - s) for _, _, _, p, s, _, _, _ in res])
print(f"\nMAE start = {mae_s:.2f} s   |   MAE end = {mae_e:.2f} s\n")
pathlib.Path("../plots").mkdir(exist_ok=True)
# ---------- 5. вывод + графики ---------------------------------------
for key, sp, st, ep, et, p, ts, te in res:
    print(f"{key:38}  start {sp:4}/{st:4}  |  end {ep:4}/{et:4}")
    # --- график -------------------------------------------------------
    plt.figure(figsize=(10, 2))
    t = np.arange(len(p))
    plt.plot(t, p, label='p(t)', lw=1)
    plt.axhline(ts, color='C2', ls='--', label='thr_start')
    plt.axhline(te, color='C3', ls='--', label='thr_end')
    plt.axvline(sp, color='C2')
    plt.axvline(st, color='C2', ls=':')
    plt.axvline(ep, color='C3')
    plt.axvline(et, color='C3', ls=':')
    plt.title(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{key}.png", dpi=120)
    plt.close()
