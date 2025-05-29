# train_transformer.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import VideoWindowDataset
from model import ClipAttention60


def main():
    train_ds = VideoWindowDataset(split="train video", augment=True)
    val_ds = VideoWindowDataset(split="val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClipAttention60().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.BCELoss()

    def step(loader, train=True):
        if train:
            model.train();
            optim.zero_grad()
        else:
            model.eval()
        tot, n = 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                loss = criterion(p, y)
                if train:
                    loss.backward();
                    optim.step();
                    optim.zero_grad()
                tot += loss.item() * x.size(0);
                n += x.size(0)
        return tot / n

    from pathlib import Path
    Path("../weights").mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for epoch in range(1, 17):
        tr = step(train_loader, True)
        vl = step(val_loader, False)
        print(f"E{epoch:02} train video={tr:.4f} val={vl:.4f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(),
                       "../weights/best_clip_attention60.pth")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()  # на случай build-exe; безопасно оставить
    main()
