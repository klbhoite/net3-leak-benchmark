"""Deep models used in the comparison (PyTorch)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNN(nn.Module):
    """Baseline: parallel 1D/2D CNN with feature fusion. A re-implementation of the
    parallel-branch idea of Bhoite & Bhosle (2026) using standard batch
    normalisation and a single fused classifier head (the modified BN, mixed
    pooling and GhostNet/soft-voting components of that work are not reproduced)."""
    def __init__(self, c_in, w):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(c_in, 64, 5, padding=2), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.AdaptiveMaxPool1d(1))
        self.b2 = nn.Sequential(
            nn.Conv2d(1, 32, (5, 3), padding=(2, 1)), nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.AdaptiveMaxPool2d(1))
        self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):                      # x: [B, W, C]
        a = self.b1(x.transpose(1, 2)).flatten(1)
        b = self.b2(x.unsqueeze(1)).flatten(1)
        return self.head(torch.cat([a, b], 1)).squeeze(1)


class CNN1D(nn.Module):
    def __init__(self, c_in, w):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c_in, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.f(x.transpose(1, 2)).flatten(1)).squeeze(1)


class CNNBiLSTM(nn.Module):
    def __init__(self, c_in, w, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(h)
        z = torch.cat([o[:, -1], o.mean(1)], 1)
        return self.head(z).squeeze(1)


class _TBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, p=0.1):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_in, c_out, k, dilation=d))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_out, c_out, k, dilation=d))
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        h = self.drop(F.relu(self.c2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + self.down(x))


class TCN(nn.Module):
    """Temporal convolutional network (causal dilated residual blocks, Bai et al. 2018)."""
    def __init__(self, c_in, w, ch=64, levels=4, k=3):
        super().__init__()
        layers, c = [], c_in
        for i in range(levels):
            layers.append(_TBlock(c, ch, k, 2 ** i)); c = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(2 * ch, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h[:, :, -1], h.mean(2)], 1)).squeeze(1)


class TransformerClf(nn.Module):
    def __init__(self, c_in, w, d=64, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(c_in, d)
        pe = torch.zeros(w, d); pos = torch.arange(w).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 1))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pe)
        return self.head(torch.cat([h[:, -1], h.mean(1)], 1)).squeeze(1)


class ConvAE(nn.Module):
    """1D convolutional autoencoder trained on leak-free windows only."""
    def __init__(self, c_in, w):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(c_in, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv1d(32, c_in, 5, padding=2))

    def forward(self, x):
        return self.dec(self.enc(x.transpose(1, 2))).transpose(1, 2)


DEEP = {"P-CNN (baseline)": PCNN, "1D-CNN": CNN1D, "CNN-BiLSTM": CNNBiLSTM,
        "TCN": TCN, "Transformer": TransformerClf}
