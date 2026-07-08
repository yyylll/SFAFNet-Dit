#!/usr/bin/env python3
"""
Low-level building blocks for SFAFNet-DiT
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_groups(cout: int) -> int:
    for g in (8, 4, 2, 1):
        if cout % g == 0:
            return g
    return 1


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, d=1, act="leaky"):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d)
        self.n = nn.GroupNorm(_safe_groups(cout), cout)
        self.a = nn.SiLU(inplace=True) if act == "silu" else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.a(self.n(self.c(x)))


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // r), nn.SiLU(), nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        z = x.mean(dim=(2, 3))
        w = self.fc(z).unsqueeze(-1).unsqueeze(-1)
        return x * w


class InterferenceLocalizationHead(nn.Module):
    def __init__(self, in_ch: int, mid: int | None = None):
        super().__init__()
        mid = mid or max(16, in_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, mid // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid // 2, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.0)  

    def forward(self, feats):
        return torch.sigmoid(self.net(feats))


class REMHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, in_ch // 2), ConvBNAct(in_ch // 2, in_ch // 4),
            nn.Conv2d(in_ch // 4, out_ch, 3, padding=1),
        )

    def forward(self, feats):
        return self.net(feats)


@torch.no_grad()
def peak_nms(heatmap: torch.Tensor, tau_conf: float = 0.5, radius: int = 3, max_sources: int = 16):
    B, _, H, W = heatmap.shape
    k = 2 * radius + 1
    pooled = F.max_pool2d(heatmap, k, stride=1, padding=radius)
    is_peak = (heatmap == pooled) & (heatmap >= tau_conf)
    out = []
    for b in range(B):
        ys, xs = torch.where(is_peak[b, 0])
        if ys.numel() == 0:
            out.append(torch.zeros(0, 3, device=heatmap.device)); continue
        s = heatmap[b, 0, ys, xs]
        o = torch.argsort(s, descending=True)[:max_sources]
        out.append(torch.stack([ys[o].float(), xs[o].float(), s[o]], 1))
    return out

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor | None = None) -> torch.Tensor:
        if cond_feat is not None:
            x = x + cond_feat
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SFAFBlock(nn.Module):
    def __init__(self, ch: int, dilations=(1, 2, 4)):
        super().__init__()
        self.dil = nn.ModuleList([ConvBNAct(ch, ch, d=r, act="silu") for r in dilations])
        self.attn = ChannelAttention(ch)
        self.fuse = ConvBNAct(ch, ch, act="silu")

    def forward(self, x, cond_feat=None):
        if cond_feat is not None:
            x = x + cond_feat         
        h = x
        for layer in self.dil:
            h = h + layer(h)          
        h = self.attn(h)
        return self.fuse(h)

    @torch.no_grad()
    def load_body_from_shared_encoder(self, enc):
        for a, b in zip(self.dil, enc.dil):
            a.load_state_dict(b.state_dict())
        self.attn.load_state_dict(enc.attn.state_dict())
        self.fuse.load_state_dict(enc.fuse.state_dict())


class ResCompress(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_safe_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_safe_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim=256, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x_seq):
        h = self.norm1(x_seq)
        x_seq = x_seq + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x_seq)
        return x_seq + self.mlp(h)


class ConditionEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, channels: tuple = (96, 192, 384, 512)):
        super().__init__()
        self.init = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(channels)):
            self.blocks.append(nn.Sequential(
                nn.GroupNorm(min(8, channels[i]), channels[i]),
                nn.SiLU(),
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
                nn.GroupNorm(min(8, channels[i]), channels[i]),
                nn.SiLU(),
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
            ))
            if i < len(channels) - 1:
                self.downs.append(nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1))

    def forward(self, cond: torch.Tensor) -> list:
        x = self.init(cond)
        features = []
        for i, block in enumerate(self.blocks):
            x = x + block(x)
            features.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return features


class ZeroInitProjection(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

