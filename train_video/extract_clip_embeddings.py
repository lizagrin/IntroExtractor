#!/usr/bin/env python
"""
Reproduce section 3.1–3.2 of the paper:
* 1 FPS frames (already extracted)        --> data/frames/{show}/{episode}/XXXXX.jpg
* Resize to 224×224,  ImageNet mean/std
* CLIP ViT-B/32  --> 512-D embeddings
* Pack sliding windows of 60 frames (stride = 60)
  into an array: (N, 60, 512) and save as .npz
"""
import os, pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import open_clip

# ---------- 1.  CLIP initialisation ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=device
)
model.eval()

# Paper-exact preprocessing: Resize only (no CenterCrop)
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ---------- 2.  Paths ----------
ROOT_FRAMES = pathlib.Path("../data/frames")
ROOT_OUT = pathlib.Path("../data/clip_windows")
ROOT_OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128  # images per forward pass
WINDOW_SIZE = 60  # seconds (frames) per window
WINDOW_STRIDE = 60  # slide step; paper uses non-overlapping windows

# ---------- 3.  Iterate over episodes ----------
for ep_dir in sorted(ROOT_FRAMES.rglob("*")):
    if not ep_dir.is_dir():
        continue

    rel_ep = ep_dir.relative_to(ROOT_FRAMES)  # show1/episode-02
    out_file = ROOT_OUT / (str(rel_ep).replace(os.sep, "_") + "_windows.npz")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        print(f"[✓] {rel_ep} — уже готово")
        continue

    frame_paths = sorted(ep_dir.glob("*.jpg"))
    if len(frame_paths) < WINDOW_SIZE:
        print(f"[!] {rel_ep}: меньше 60 кадров, пропуск")
        continue

    # ---------- 3a.  Encode all frames ----------
    embeddings = np.empty((len(frame_paths), 512), dtype=np.float32)

    batch_imgs, idx_buf = [], []
    for idx, fp in enumerate(tqdm(frame_paths, desc=str(rel_ep), unit="f")):
        img = preprocess(Image.open(fp)).unsqueeze(0)  # (1,3,224,224)
        batch_imgs.append(img)
        idx_buf.append(idx)

        if len(batch_imgs) == BATCH_SIZE or idx == len(frame_paths) - 1:
            with torch.no_grad():
                feats = model.encode_image(torch.cat(batch_imgs).to(device))
            embeddings[idx_buf] = feats.cpu().float().numpy()
            batch_imgs, idx_buf = [], []

    # ---------- 3b.  Build sliding windows ----------
    starts = range(0, len(embeddings) - WINDOW_SIZE + 1, WINDOW_STRIDE)
    windows = np.stack([embeddings[s:s + WINDOW_SIZE] for s in starts])  # (N,60,512)

    # ---------- 3c.  Save ----------
    np.savez_compressed(out_file, windows=windows, start_indices=np.array(list(starts)))
    print(f"[✓] {rel_ep}: сохранено {windows.shape[0]} окон → {out_file}")

print("CLIP-окна готовы — можно тренировать трансформер!")
