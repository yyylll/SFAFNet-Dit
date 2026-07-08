#!/usr/bin/env python3
"""TF total-first SFAFNet (two-decoder): shared encoder -> R-decoder + decomp-decoder -> heads."""
from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import (SFAFBlock, ResBlock, ResCompress, _TransformerBlock,
                     ConditionEncoder, ZeroInitProjection, REMHead, InterferenceLocalizationHead)


def _mk(block, ch):
    if block == "resblock":
        return ResBlock(ch, ch)
    return SFAFBlock(ch)


class SharedEncoder(nn.Module):
    def __init__(self, in_ch=5, channels=(64, 128, 256, 384), n_per_level=2,
                 bottleneck_dim=256, n_transformer=4, n_heads=4, block="sfaf"):
        super().__init__()
        self.channels = channels
        self.block = block
        self.cond_encoder = ConditionEncoder(in_ch, channels)
        self.cond_projs = nn.ModuleList([ZeroInitProjection(c) for c in channels])
        self.init_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(channels):
            self.enc_blocks.append(nn.ModuleList([_mk(block, ch) for _ in range(n_per_level)]))
            if i < len(channels) - 1:
                self.downs.append(nn.Conv2d(ch, channels[i + 1], 4, stride=2, padding=1))
        self.bottleneck_sfaf = _mk(block, channels[-1])
        self.bottleneck_in = nn.Conv2d(channels[-1], bottleneck_dim, 1)
        self.transformers = nn.ModuleList([_TransformerBlock(bottleneck_dim, n_heads) for _ in range(n_transformer)])
        self.bottleneck_out = nn.Conv2d(bottleneck_dim, channels[-1], 1)

    def forward(self, x):
        cond_projs = [proj(c) for proj, c in zip(self.cond_projs, self.cond_encoder(x))]
        h = self.init_conv(x)
        skips = []
        for i, blocks in enumerate(self.enc_blocks):
            h = blocks[0](h, cond_feat=cond_projs[i])
            for blk in blocks[1:]:
                h = blk(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.bottleneck_sfaf(h)
        h = self.bottleneck_in(h)
        B, C, Hh, Ww = h.shape
        seq = h.flatten(2).transpose(1, 2)
        for tb in self.transformers:
            seq = tb(seq)
        h = self.bottleneck_out(seq.transpose(1, 2).view(B, C, Hh, Ww))
        return h, skips, cond_projs


class SFAFDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 384), n_per_level=2, block="sfaf"):
        super().__init__()
        self.channels = channels
        self.ups = nn.ModuleList()
        self.dec_compress = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(channels))):
            ch = channels[i]
            self.dec_compress.append(ResCompress(2 * ch, ch))
            self.dec_blocks.append(nn.ModuleList([_mk(block, ch) for _ in range(max(1, n_per_level - 1))]))
            if i > 0:
                self.ups.append(nn.ConvTranspose2d(ch, channels[i - 1], 4, stride=2, padding=1))

    def forward(self, h, skips, cond_projs):
        for i, (compress, blocks) in enumerate(zip(self.dec_compress, self.dec_blocks)):
            level = len(self.channels) - 1 - i
            h = compress(torch.cat([h, skips[level]], dim=1))
            for blk in blocks:
                h = blk(h, cond_feat=cond_projs[level])
            if i < len(self.ups):
                h = self.ups[i](h)
        return h

class RHead(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.head = REMHead(ch, 1)

    def forward(self, feat):
        return self.head(feat)


class DecompHeadsTF(nn.Module):
    def __init__(self, ch: int, rss_center: float = -57.6, rss_scale: float = 8.52,
                 sinr_center: float = 0.0, sinr_scale: float = 12.0, noise_floor_dbm: float = -95.0):
        super().__init__()
        self.s_head = REMHead(ch, 1)
        self.i_head = REMHead(ch, 1)
        self.var_head = REMHead(ch, 1)
        self.loc_head = InterferenceLocalizationHead(ch)
        for nm, v in [("rss_center", rss_center), ("rss_scale", rss_scale),
                      ("sinr_center", sinr_center), ("sinr_scale", sinr_scale),
                      ("noise_floor_dbm", noise_floor_dbm)]:
            self.register_buffer(nm, torch.tensor(float(v)))

    def _denorm(self, x):
        return x * self.rss_scale + self.rss_center

    def combine(self, S_n, I_n):
        s = torch.pow(10.0, self._denorm(S_n) / 10.0); i = torch.pow(10.0, self._denorm(I_n) / 10.0)
        return (10.0 * torch.log10((s + i).clamp(min=1e-30)) - self.rss_center) / self.rss_scale

    def sinr(self, S_n, I_n):
        s = torch.pow(10.0, self._denorm(S_n) / 10.0); i = torch.pow(10.0, self._denorm(I_n) / 10.0)
        n = torch.pow(10.0, self.noise_floor_dbm / 10.0)
        return (10.0 * torch.log10((s / (i + n)).clamp(min=1e-30)) - self.sinr_center) / self.sinr_scale

    def forward(self, feat_decomp, s_prior=None):
        S = self.s_head(feat_decomp)
        if s_prior is not None:
            S = S + s_prior
        I = self.i_head(feat_decomp)
        return {
            "S": S, "I": I,
            "R_phys": self.combine(S, I),
            "SINR": self.sinr(S, I),
            "H_IN": self.loc_head(feat_decomp),
            "logvar": self.var_head(feat_decomp),
        }


class DecompSFAFNetTF(nn.Module):
    def __init__(self, ch: int = 64, channels=None, n_per_level=2, block="sfaf",
                 sparse_ch=2, building_ch=1, tx_ch=2,
                 rss_center: float = -57.6, rss_scale: float = 8.52,
                 sinr_center: float = 0.0, sinr_scale: float = 12.0, noise_floor_dbm: float = -95.0):
        super().__init__()
        if channels is None:
            channels = (ch, 2 * ch, 4 * ch, 6 * ch)
        in_ch = sparse_ch + building_ch + tx_ch
        self.encoder = SharedEncoder(in_ch=in_ch, channels=channels, n_per_level=n_per_level, block=block)
        self.R_decoder = SFAFDecoder(channels=channels, n_per_level=n_per_level, block=block)
        self.r_head = RHead(channels[0])
        self.decomp_decoder = SFAFDecoder(channels=channels, n_per_level=n_per_level, block=block)
        self.decomp_heads = DecompHeadsTF(channels[0], rss_center=rss_center, rss_scale=rss_scale,
                                          sinr_center=sinr_center, sinr_scale=sinr_scale, noise_floor_dbm=noise_floor_dbm)
        for nm in ("rss_center", "rss_scale", "sinr_center", "sinr_scale", "noise_floor_dbm"):
            setattr(self, nm, getattr(self.decomp_heads, nm))

    def forward(self, sparse_obs, sparse_mask, building_mask, tx_feature_map, s_prior=None, decomp=True):
        x = torch.cat([sparse_obs, sparse_mask, building_mask, tx_feature_map], dim=1)
        h, skips, cond_projs = self.encoder(x)
        R_direct = self.r_head(self.R_decoder(h, skips, cond_projs))
        if not decomp:
            return {"R_direct": R_direct, "R_raw": R_direct}
        out = self.decomp_heads(self.decomp_decoder(h, skips, cond_projs), s_prior=s_prior)
        out["R_direct"] = R_direct
        out["R_raw"] = R_direct
        return out

