#!/usr/bin/env python3
"""Minimal self-contained forward-pass demo of the SFAFNet-DiT core modules """
import torch
from model import DecompSFAFNetTF, PhysicsDiT, DiffusionSchedule, PhysicsPriorEncoder
from losses_tf import TotalFirstLoss
import physics_prior as PP

torch.manual_seed(0)
B, H, W = 2, 128, 128
obs  = torch.randn(B, 1, H, W)
mask = (torch.rand(B, 1, H, W) > 0.9).float()
bld  = (torch.rand(B, 1, H, W) > 0.8).float()
txm  = torch.zeros(B, 2, H, W)

# Stage 1: total-first dual-decoder SFAFNet -> total field + (S, I, SINR) decomposition + localization.
stage1 = DecompSFAFNetTF(ch=32, block="sfaf").eval()
with torch.no_grad():
    o = stage1(obs, mask, bld, txm, s_prior=torch.zeros(B, 1, H, W))
print("[Stage-1] outputs:")
for k in ("R_direct", "S", "I", "SINR", "H_IN", "logvar"):
    print(f"    {k:9s} {tuple(o[k].shape)}")

# PCDL objective: L_R (total) plus the decomposition group with its power-consistency anchor.
crit = TotalFirstLoss()
batch = {"rem": torch.randn(B, 1, H, W), "building_mask": bld,
         "gt_signal": torch.randn(B, 1, H, W), "gt_interf": torch.randn(B, 1, H, W),
         "gt_sinr": torch.randn(B, 1, H, W),
         "gt_interference_positions": [torch.tensor([[40.0, 60.0]]) for _ in range(B)]}
L_R, L_dec, parts = crit(o, batch)
print(f"[PCDL]    L_R={float(L_R):.4f}  L_decomp_group={float(L_dec):.4f}")

# Physics-prior conditioning map M_phy: decomposition + interference field + Helmholtz curvature.
with torch.no_grad():
    k2 = PP.helmholtz_k2(o["R_direct"], bld)
    M_phy = PhysicsPriorEncoder()(o["S"], o["I"], o["SINR"], bld, o["I"].abs(), k2=k2)
print(f"[M_phy]   physics-prior conditioning map {tuple(M_phy.shape)}")

# Stage 2: decomposition-conditioned Physics-DiT refiner (one denoising step).
dit   = PhysicsDiT(mphy_ch=M_phy.shape[1], img=H).eval()
sched = DiffusionSchedule(T=1000)
with torch.no_grad():
    t   = torch.randint(0, sched.T, (B,))
    x_t = sched.q_sample(o["R_direct"], t, torch.randn(B, 1, H, W))
    observed = PhysicsDiT.build_cond_input(o["R_direct"], obs, mask)
    eps_pred = dit(x_t, t, o["R_direct"], M_phy, observed)
print(f"[Stage-2] Physics-DiT eps_pred {tuple(eps_pred.shape)}")

