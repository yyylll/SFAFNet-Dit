#!/usr/bin/env python3
"""2-task Conflict-Averse Gradient (CAGrad)."""
import torch


@torch.no_grad()
def cagrad_two(g1, g2, c=0.5, steps=101, eps=1e-8):
    """g1, g2: flat 1-D gradient tensors (same shape)."""
    g0 = 0.5 * (g1 + g2)
    g0n = g0.norm() + eps
    ws = torch.linspace(0.0, 1.0, steps, device=g1.device, dtype=g1.dtype)
    best_w, best_F = 0.5, None
    for w in ws:
        gw = w * g1 + (1.0 - w) * g2
        F = float(g0 @ gw) + c * float(g0n) * float(gw.norm() + eps)
        if best_F is None or F < best_F:
            best_F = F; best_w = float(w)
    gw = best_w * g1 + (1.0 - best_w) * g2
    return g0 + (c * g0n / (gw.norm() + eps)) * gw
