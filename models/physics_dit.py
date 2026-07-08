#!/usr/bin/env python3
"""
Physics-DiT (conditional diffusion Transformer). 
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbed(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float().unsqueeze(1) * f.unsqueeze(0)
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class DiTBlock(nn.Module):
    def __init__(self, dim=256, heads=4, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim); self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim)); nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond_tokens, t_emb):
        g1, g2, g3 = self.ada(t_emb).chunk(3, dim=1)
        x = x + g1.unsqueeze(1) * self.sa(self.n1(x), self.n1(x), self.n1(x), need_weights=False)[0]
        x = x + g2.unsqueeze(1) * self.ca(self.n2(x), cond_tokens, cond_tokens, need_weights=False)[0]
        x = x + g3.unsqueeze(1) * self.ff(self.n3(x))
        return x

class PhysicsDiT(nn.Module):
    def __init__(self, mphy_ch=16, dim=256, depth=6, heads=4, patch=8, img=128):
        super().__init__()
        self.patch = patch; self.dim = dim; self.img = img
        n_tokens = (img // patch) ** 2
        self.in_proj = nn.Conv2d(3, dim, patch, stride=patch)
        self.cond_proj = nn.Conv2d(mphy_ch, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.cpos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.time = TimeEmbed(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, patch * patch)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def build_cond_input(R_raw, R_S, M_mask, v_fill=0.0):
        observed = M_mask * R_S + (1.0 - M_mask) * v_fill
        return observed

    def forward(self, x_t, t, R_raw, M_phy, observed):
        B = x_t.shape[0]; p = self.patch; n = self.img // p
        xin = torch.cat([x_t, R_raw, observed], dim=1)
        tok = self.in_proj(xin).flatten(2).transpose(1, 2) + self.pos
        ctok = self.cond_proj(M_phy).flatten(2).transpose(1, 2) + self.cpos
        te = self.time(t)
        for blk in self.blocks:
            tok = blk(tok, ctok, te)
        eps = self.out_proj(self.out_norm(tok))
        eps = eps.transpose(1, 2).reshape(B, 1, p, p, n, n)
        eps = eps.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, self.img, self.img)
        return eps

class DiffusionSchedule:
    def __init__(self, T=1000, device="cpu"):
        s = 0.008
        ts = torch.linspace(0, T, T + 1, device=device) / T
        ac = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.betas = (1 - ac[1:] / ac[:-1]).clamp(1e-4, 0.999)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.T = T

    def q_sample(self, R, t, eps):
        ab = self.abar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * R + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def p_step(self, model, x_t, t, R_raw, M_phy, observed):
        a = self.alphas[t].view(-1, 1, 1, 1); ab = self.abar[t].view(-1, 1, 1, 1)
        eps = model(x_t, t, R_raw, M_phy, observed)
        mean = (x_t - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
        if int(t[0]) == 0:
            return mean
        z = torch.randn_like(x_t)
        return mean + self.betas[t].view(-1, 1, 1, 1).sqrt() * z
    
    @torch.no_grad()
    def ddim_sample(self, model, x, R_raw, M_phy, observed, steps, eta=0.0, x0_clamp=10.0):
        B = x.shape[0]
        ts = torch.linspace(self.T - 1, 0, steps + 1, device=x.device).round().long()
        for i in range(steps):
            t = ts[i].expand(B)
            tp = int(ts[i + 1])
            ab = self.abar[ts[i]].view(1, 1, 1, 1)
            ab_p = (self.abar.new_ones(()) if tp == 0 else self.abar[tp]).view(1, 1, 1, 1)
            eps = model(x, t, R_raw, M_phy, observed)
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-4)
            if x0_clamp is not None:
                x0 = x0.clamp(-x0_clamp, x0_clamp)
            if eta > 0 and tp > 0:
                sigma = eta * ((1 - ab_p) / (1 - ab)).sqrt() * (1 - ab / ab_p).clamp(min=0).sqrt()
                x = ab_p.sqrt() * x0 + (1 - ab_p - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * torch.randn_like(x)
            else:
                x = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps
        return x

    def ddim_sample_dps(self, model, R_raw, M_phy, observed, R_S, M, steps, t0=400,
                        scale=1.0, x0_clamp=10.0, weight=None, sigma_guide=None):
        was = getattr(model, "training", False)
        if hasattr(model, "eval"):
            model.eval()
        B = R_raw.shape[0]; dev = R_raw.device
        t0i = min(int(t0), self.T - 1)
        ab0 = self.abar[t0i].view(1, 1, 1, 1)
        x = (ab0.sqrt() * R_raw + (1 - ab0).sqrt() * torch.randn_like(R_raw)).detach()
        ts = torch.linspace(t0i, 0, steps + 1, device=dev).round().long()
        wsqrt = weight.sqrt() if weight is not None else None
        try:
            for i in range(steps):
                t = ts[i].expand(B); tp = int(ts[i + 1])
                ab = self.abar[ts[i]].view(1, 1, 1, 1)
                ab_p = (self.abar.new_ones(()) if tp == 0 else self.abar[tp]).view(1, 1, 1, 1)
                x_t = x.detach().requires_grad_(True)
                with torch.enable_grad():
                    eps = model(x_t, t, R_raw, M_phy, observed)
                    x0 = (x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-4)
                    if x0_clamp is not None:
                        x0 = x0.clamp(-x0_clamp, x0_clamp)
                    resid = M * (R_S - x0)
                    if wsqrt is not None:
                        resid = wsqrt * resid
                    norm = torch.linalg.norm(resid)
                g = torch.autograd.grad(norm, x_t)[0]
                x_ddim = (ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps).detach()
                x = (x_ddim - scale * g).detach()
                if sigma_guide is not None:
                    x_known = ab_p.sqrt() * R_raw + (1 - ab_p).sqrt() * torch.randn_like(R_raw)
                    x = (sigma_guide * x + (1.0 - sigma_guide) * x_known).detach()
            return x
        finally:
            if was and hasattr(model, "train"):
                model.train()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    model = PhysicsDiT(mphy_ch=16, dim=128, depth=2, heads=4, patch=8, img=H)
    sched = DiffusionSchedule(T=1000)
    R = torch.randn(B, 1, H, W)
    R_raw = torch.randn(B, 1, H, W); M_phy = torch.randn(B, 16, H, W)
    R_S = torch.randn(B, 1, H, W); M_mask = (torch.rand(B, 1, H, W) > 0.97).float()
    observed = PhysicsDiT.build_cond_input(R_raw, R_S, M_mask)
    t = torch.randint(0, 1000, (B,))
    eps = torch.randn_like(R)
    x_t = sched.q_sample(R, t, eps)
    pred = model(x_t, t, R_raw, M_phy, observed)
    print("x_t:", tuple(x_t.shape), "| eps_pred:", tuple(pred.shape))
    assert pred.shape == (B, 1, H, W)
    obs_uses_only_sparse = torch.allclose(observed * (1 - M_mask), torch.zeros_like(observed))
    print("conditional 'observed' nonzero only at sensors (no dense GT leak):", obs_uses_only_sparse)
    assert obs_uses_only_sparse
    x_prev = sched.p_step(model, x_t, t, R_raw, M_phy, observed)
    assert x_prev.shape == (B, 1, H, W)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
