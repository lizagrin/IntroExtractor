# src/datasets.py
import numpy as np, torch, pathlib, pandas as pd


class VideoWindowDataset(torch.utils.data.Dataset):
    def __init__(self, root_npz="../data/clip_windows", labels_csv="../data/labels.csv",
                 split="train_video", test_shows=("show2",), augment=False):
        self.root = pathlib.Path(root_npz)
        self.labels_df = pd.read_csv(labels_csv)

        # 2.1. строим словарь  «эпизод → (start_sec, end_sec)»
        self.meta = {}
        for _, row in self.labels_df.iterrows():
            start = self._hhmmss_to_sec(row["start_main"])
            end = self._hhmmss_to_sec(row["end_main"])
            self.meta[pathlib.Path(row["file"]).stem] = (start, end)

        # 2.2. читаем все npz
        self.items = []
        for p in self.root.glob("*.npz"):
            show = p.name.split("_")[0]  # show1 / show2
            if (split == "train_video" and show in test_shows) or \
                    (split == "val" and show not in test_shows):
                continue
            npz = np.load(p)
            win = npz["windows"]  # (M,60,512)
            idxs = npz["start_indices"]  # (M,)
            video_key = "_".join(p.stem.split("_")[1:-1])  # оригинальное имя без _windows
            s, e = self.meta[video_key]

            # формируем метки
            for w, start_t in zip(win, idxs):
                t = np.arange(start_t, start_t + 60)  # 60 секунд
                y = ((t < s) | (t >= e)).astype(np.float32)  # 1-для intro/credits
                self.items.append((w.astype(np.float32), y))

        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        if self.augment:
            x, y = self._augment(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)

    # ---------- utils ----------
    @staticmethod
    def _hhmmss_to_sec(ts):
        h, m, s = map(float, ts.split(":"))
        return int(h * 3600 + m * 60 + s)

    def _augment(self, x, y):
        # a) random temporal shift ±5 s (см. §4.1.5)
        shift = np.random.randint(-5, 6)
        if shift != 0:
            x = np.roll(x, shift, axis=0)
            y = np.roll(y, shift)
        # b) frame-substitution 10–30 %
        mask = np.random.rand(60) < np.random.uniform(0.1, 0.3)
        x[mask] = x[np.random.permutation(60)[mask]]
        return x, y
