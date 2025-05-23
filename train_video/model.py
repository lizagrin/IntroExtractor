# src/model.py
import torch, torch.nn as nn


class ClipAttention60(nn.Module):
    def __init__(self, d_model=512, nhead=16, nlayers=16, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(60, d_model))  # learnable PE (§3.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # 60 независимых классификаторов (§3.5)
        self.classifiers = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(60)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B,60,512)
        x = x + self.pos  # добавить позиционку
        h = self.encoder(x)  # (B,60,512)

        outs = []
        for t in range(60):
            outs.append(self.classifiers[t](h[:, t, :]))  # (B,1)
        logits = torch.cat(outs, dim=1)  # (B,60)
        return self.sigmoid(logits)  # σ → вероятность «intro/credits»
