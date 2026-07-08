#!/usr/bin/env python3
"""Physics prior M_phy construction."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _los_blockage_distance(bx, by, building, n_override=None):
    H, W = building.shape
    device = building.device
    gy = torch.arange(H, device=device).view(H, 1).expand(H, W).float()
    gx = torch.arange(W, device=device).view(1, W).expand(H, W).float()
    n = int(n_override) if n_override else int(max(H, W))
    ts = torch.linspace(0, 1, n, device=device).view(n, 1, 1)
    py = by + (gy.unsqueeze(0) - by) * ts
    px = bx + (gx.unsqueeze(0) - bx) * ts
    iy = py.round().long().clamp(0, H - 1)
    ix = px.round().long().clamp(0, W - 1)
    hits = building[iy, ix]
    return hits.mean(0)


def ray_traced_primary_field(tx_positions, tx_powers, building_mask,
                             alpha_los=2.0, beta=-30.0, gamma=20.0, dxk=1.0,
                             clamp_max=None, n_samples=None):
    B, _, H, W = building_mask.shape
    device = building_mask.device
    out = torch.full((B, 1, H, W), -120.0, device=device)
    ys = torch.arange(H, device=device).view(H, 1).float()
    xs = torch.arange(W, device=device).view(1, W).float()
    for b in range(B):
        bmask = building_mask[b, 0]
        pos = tx_positions[b]; pw = tx_powers[b]
        if torch.is_tensor(pos) is False:
            pos = torch.as_tensor(pos, device=device, dtype=torch.float32)
        acc = torch.zeros(H, W, device=device)
        for l in range(pos.shape[0]):
            by, bx = pos[l, 0].item(), pos[l, 1].item()
            d = torch.sqrt((ys - by) ** 2 + (xs - bx) ** 2) / dxk + 1e-3
            dD = _los_blockage_distance(bx, by, bmask, n_override=n_samples) * d
            G = beta - 10 * alpha_los * torch.log10(d) - gamma * dD
            P = float(pw[l]) if torch.is_tensor(pw) or isinstance(pw, (list, tuple)) else float(pw)
            acc = acc + 10 ** ((P + G) / 10.0)
        out[b, 0] = 10 * torch.log10(acc + 1e-12)
    if clamp_max is not None:
        out = out.clamp(max=float(clamp_max))
    return out


def fspl_primary_field(tx_positions, tx_powers, tx_heights, building_mask,
                       pixel_size_m=5.0, rx_height_m=1.5, freq_hz=5.75e9):
    B, _, H, W = building_mask.shape
    device = building_mask.device
    log4pi = 20.0 * math.log10(4.0 * math.pi / (299792458.0 / freq_hz))
    ys = torch.arange(H, device=device).view(H, 1).float()
    xs = torch.arange(W, device=device).view(1, W).float()
    out = torch.full((B, 1, H, W), -150.0, device=device)
    for b in range(B):
        pos = tx_positions[b]; pw = tx_powers[b]; ph = tx_heights[b]
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos, device=device, dtype=torch.float32)
        if pos.numel() == 0:
            continue
        acc = torch.zeros(H, W, device=device)
        for l in range(pos.shape[0]):
            by, bx = float(pos[l, 0]), float(pos[l, 1])
            h = float(ph[l]) if torch.is_tensor(ph) or isinstance(ph, (list, tuple)) else float(ph)
            P = float(pw[l]) if torch.is_tensor(pw) or isinstance(pw, (list, tuple)) else float(pw)
            d_h = torch.sqrt((ys - by) ** 2 + (xs - bx) ** 2) * pixel_size_m
            d_3d = torch.sqrt(d_h ** 2 + (h - rx_height_m) ** 2).clamp(min=0.01)
            fspl = 20.0 * torch.log10(d_3d) + log4pi
            acc = acc + torch.pow(10.0, (P - fspl) / 10.0)
        out[b, 0] = 10.0 * torch.log10(acc.clamp(min=1e-15))
    return out


def fspl_interference_field(q_hat_list, power_list, building_mask,
                            h_nominal=30.0, pixel_size_m=5.0, rx_height_m=1.5, freq_hz=5.75e9):
    B = len(q_hat_list)
    device = building_mask.device
    _, _, H, W = building_mask.shape
    log4pi = 20.0 * math.log10(4.0 * math.pi / (299792458.0 / freq_hz))
    ys = torch.arange(H, device=device).view(H, 1).float()
    xs = torch.arange(W, device=device).view(1, W).float()
    out = torch.full((B, 1, H, W), -150.0, device=device)
    for b in range(B):
        q = q_hat_list[b]; pw = power_list[b]
        if q.numel() == 0:
            continue
        acc = torch.zeros(H, W, device=device)
        for k in range(q.shape[0]):
            yk = float(q[k, 0]); xk = float(q[k, 1])
            d_h = torch.sqrt((ys - yk) ** 2 + (xs - xk) ** 2) * pixel_size_m
            d_3d = torch.sqrt(d_h ** 2 + (h_nominal - rx_height_m) ** 2).clamp(min=0.01)
            fspl = 20.0 * torch.log10(d_3d) + log4pi
            acc = acc + torch.pow(10.0, (pw[k] - fspl) / 10.0)
        out[b, 0] = 10.0 * torch.log10(acc.clamp(min=1e-15))
    return out


def interference_field_from_estimates(q_hat_list, H, W, sigma=4.0, device=None):
    B = len(q_hat_list)
    device = device or (q_hat_list[0].device if len(q_hat_list[0]) else "cpu")
    ys = torch.arange(H, device=device).view(1, H, 1).float()
    xs = torch.arange(W, device=device).view(1, 1, W).float()
    out = torch.zeros(B, 1, H, W, device=device)
    two_s2 = 2.0 * sigma * sigma
    for b in range(B):
        q = q_hat_list[b]
        if q.numel() == 0:
            continue
        gy = q[:, 0].view(-1, 1, 1); gx = q[:, 1].view(-1, 1, 1)
        sc = q[:, 2].view(-1, 1, 1)
        g = sc * torch.exp(-((ys - gy) ** 2 + (xs - gx) ** 2) / two_s2)
        out[b, 0] = g.sum(0)
    return out


def interference_field_from_heatmap(heatmap, sigma=2.0, scale=1.0):
    ks = max(3, int(2 * round(2 * sigma) + 1))
    dev = heatmap.device
    c = torch.arange(ks, device=dev).float() - ks // 2
    g1 = torch.exp(-(c ** 2) / (2 * sigma * sigma)); g1 = g1 / g1.sum()
    k = (g1[:, None] * g1[None, :]).view(1, 1, ks, ks)
    return scale * F.conv2d(heatmap, k, padding=ks // 2)


class PhysicsPriorEncoder(nn.Module):
    def __init__(self, out_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.GELU(),
        )

    def forward(self, L_ray, I_field, building_mask):
        x = torch.cat([L_ray, I_field, building_mask], dim=1)
        return self.net(x)

_K2_LAPLACIAN = torch.tensor([
    [0., 0., -1., 0., 0.],
    [0., 0., -4., 0., 0.],
    [-1., -4., 20., -4., -1.],
    [0., 0., -4., 0., 0.],
    [0., 0., -1., 0., 0.]]).view(1, 1, 5, 5)


def helmholtz_k2(R_raw, building_mask, eps=1e-3, q=0.99, scale=None):
    w = _K2_LAPLACIAN.to(R_raw.device, R_raw.dtype)
    L = F.conv2d(F.pad(R_raw, (2, 2, 2, 2), mode="reflect"), w)
    k2 = L / (R_raw.abs() + eps)
    if scale is None:
        scale = torch.quantile(k2.abs().flatten(1), q, dim=1).view(-1, 1, 1, 1)
    k2 = (k2 / (scale + eps)).clamp(-1.0, 1.0)
    return k2 * (building_mask < 0.5).float()


def curvature_k2_raw(R, eps=1e-3):
    w = _K2_LAPLACIAN.to(R.device, R.dtype)
    L = F.conv2d(F.pad(R, (2, 2, 2, 2), mode="reflect"), w)
    return L / (R.abs() + eps)


def curvature_consistency_loss(R_ref, R_gt, building_mask, eps=1e-3):
    k2r = curvature_k2_raw(R_ref, eps)
    k2g = curvature_k2_raw(R_gt, eps)
    valid = (building_mask < 0.5).float()
    return (((k2r - k2g) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)


if __name__ == "__main__":
    import sfafnet_dual_head as S
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    bld = (torch.rand(B, 1, H, W) > 0.85).float()
    txp = [torch.tensor([[10., 10.], [50., 40.]]), torch.tensor([[30., 30.]])]
    txpw = [torch.tensor([40., 38.]), torch.tensor([42.])]
    L_ray = ray_traced_primary_field(txp, txpw, bld)
    print("L_ray:", tuple(L_ray.shape), "range [%.1f,%.1f] dB" % (L_ray.min(), L_ray.max()))
    hm = torch.zeros(B, 1, H, W); hm[0, 0, 20, 44] = 0.9; hm[1, 0, 33, 12] = 0.8
    q_hat = S.peak_nms(hm, tau_conf=0.3)
    I_field = interference_field_from_estimates(q_hat, H, W)
    enc = PhysicsPriorEncoder(16)
    M_phy = enc(L_ray, I_field, bld)
    print("I_field:", tuple(I_field.shape), "| M_phy:", tuple(M_phy.shape))
    assert L_ray.shape == (B, 1, H, W) and M_phy.shape == (B, 16, H, W)
    assert L_ray[0].std() > 1.0, "L_ray should vary spatially"
